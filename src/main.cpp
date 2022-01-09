#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <array>
#include <memory>
#include "utils/vulkan.h"
#include "app-context/VulkanApplicationContext.h"
#include "app-context/VulkanSwapchain.h"
#include "app-context/VulkanGlobal.h"
#include "utils/RootDir.h"
#include "utils/glm.h"
#include "utils/Camera.h"
#include "scene/Mesh.h"
#include "scene/Scene.h"
#include "scene/DrawableModel.h"
#include "render-context/ForwardRenderPass.h"
#include "render-context/FlatRenderPass.h"
#include "render-context/RenderSystem.h"
#include "scene/ComputeMaterial.h"
#include "scene/ComputeModel.h"

// TODO: Organize includes!

const std::string path_prefix = std::string(ROOT_DIR) + "resources/";

float mouseOffsetX, mouseOffsetY;
void mouse_callback(GLFWwindow *window, double xpos, double ypos);
void processInput(GLFWwindow *window);

float deltaTime = 0.0f; // Time between current frame and last frame
float lastFrame = 0.0f; // Time of last frame
Camera camera(glm::vec3(0.0f, 0.0f, 0.0f));

struct UniformBufferObject
{
    glm::vec3 camPosition;
    float time;
};

class HelloComputeApplication
{
public:
    void run()
    {
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    
    std::shared_ptr<mcvkp::ComputeModel> computeModel;

    std::shared_ptr<mcvkp::Scene> postProcessScene;

    std::vector<VkCommandBuffer> commandBuffers;
    
    const int MAX_FRAMES_IN_FLIGHT = 2;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> imagesInFlight;

    // Initializing layouts and models.
    void initScene()
    {
        using namespace mcvkp;
        uint32_t descriptorSetsSize = VulkanGlobal::swapchainContext.swapChainImageViews.size();

        auto uniformBufferBundle = std::make_shared<mcvkp::BufferBundle>(descriptorSetsSize);
        BufferUtils::createBundle<UniformBufferObject>(uniformBufferBundle.get(), UniformBufferObject(),
                                                       VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);

        auto targetTexture = std::make_shared<mcvkp::Image>();
        mcvkp::ImageUtils::createImage(VulkanGlobal::swapchainContext.swapChainExtent.width,
                                       VulkanGlobal::swapchainContext.swapChainExtent.height,
                                       1,
                                       VK_SAMPLE_COUNT_1_BIT,
                                       VK_FORMAT_R8G8B8A8_UNORM,
                                       VK_IMAGE_TILING_OPTIMAL,
                                       VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                                       VK_IMAGE_ASPECT_COLOR_BIT,
                                       VMA_MEMORY_USAGE_GPU_ONLY,
                                       targetTexture);
        mcvkp::ImageUtils::transitionImageLayout(targetTexture->image,
                                                 VK_FORMAT_R8G8B8A8_UNORM,
                                                 VK_IMAGE_LAYOUT_UNDEFINED,
                                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                 1);
        vkDeviceWaitIdle(VulkanGlobal::context.device);
        auto computeMaterial = std::make_shared<ComputeMaterial>(path_prefix + "/shaders/generated/mandelbrot.spv");
        computeMaterial->addBufferBundle(uniformBufferBundle, VK_SHADER_STAGE_COMPUTE_BIT);
        computeMaterial->addStorageImage(targetTexture, VK_SHADER_STAGE_COMPUTE_BIT);

        computeModel = std::make_shared<ComputeModel>(computeMaterial);

        postProcessScene = std::make_shared<Scene>(RenderPassType::eFlat);

        auto screenTex = std::make_shared<Texture>(targetTexture);
        auto screenMaterial = std::make_shared<Material>(
            path_prefix + "/shaders/generated/post-process-vert.spv",
            path_prefix + "/shaders/generated/post-process-frag.spv");
        screenMaterial->addTexture(screenTex, VK_SHADER_STAGE_FRAGMENT_BIT);
        postProcessScene->addModel(std::make_shared<DrawableModel>(screenMaterial, MeshType::ePlane));
    }

    void updateScene(uint32_t currentImage)
    {
        float currentTime = (float)glfwGetTime();
        UniformBufferObject ubo = {camera.Position, currentTime};

        auto &allocation = computeModel->getMaterial()->getBufferBundles()[0].data->buffers[currentImage]->allocation;
        void *data;
        vmaMapMemory(VulkanGlobal::context.allocator, allocation, &data);
        memcpy(data, &ubo, sizeof(ubo));
        vmaUnmapMemory(VulkanGlobal::context.allocator, allocation);
    }

    void createCommandBuffers()
    {
        commandBuffers.resize(VulkanGlobal::context.swapChainImageCount);
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = VulkanGlobal::context.commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

        auto tagetImage = computeModel->getMaterial()->getStorageImages()[0].data;
        if (vkAllocateCommandBuffers(VulkanGlobal::context.device, &allocInfo, commandBuffers.data()) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate command buffers!");
        }

        for (size_t i = 0; i < commandBuffers.size(); i++)
        {

            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = 0;                  // Optional
            beginInfo.pInheritanceInfo = nullptr; // Optional

            if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to begin recording command buffer!");
            }

            VkImageMemoryBarrier computeMemoryBarrier = {};
            computeMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            computeMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            computeMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            computeMemoryBarrier.image = tagetImage->image;
            computeMemoryBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            computeMemoryBarrier.srcAccessMask = 0;
            computeMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;

            vkCmdPipelineBarrier(
                commandBuffers[i],
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1, &computeMemoryBarrier);

            computeModel->computeCommand(commandBuffers[i], i, tagetImage->width / 32, tagetImage->height / 32, 1);

            VkImageMemoryBarrier screenQuadMemoryBarrier = {};
            screenQuadMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            screenQuadMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            screenQuadMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            screenQuadMemoryBarrier.image = tagetImage->image;
            screenQuadMemoryBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            screenQuadMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            screenQuadMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(
                commandBuffers[i],
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1, &screenQuadMemoryBarrier);

            postProcessScene->writeRenderCommand(commandBuffers[i], i);

            if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to record command buffer!");
            }
        }
    }

    void createSyncObjects()
    {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        imagesInFlight.resize(VulkanGlobal::swapchainContext.swapChainImageViews.size());

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            if (vkCreateSemaphore(VulkanGlobal::context.device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(VulkanGlobal::context.device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(VulkanGlobal::context.device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS)
            {

                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }
    }

    size_t currentFrame = 0;
    void drawFrame()
    {
        vkWaitForFences(VulkanGlobal::context.device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(VulkanGlobal::context.device,
                                                VulkanGlobal::swapchainContext.swapChain,
                                                UINT64_MAX,
                                                imageAvailableSemaphores[currentFrame],
                                                VK_NULL_HANDLE,
                                                &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            return;
        }
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        {
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        // Check if a previous frame is using this image (i.e. there is its fence to wait on)
        if (imagesInFlight[imageIndex] != VK_NULL_HANDLE)
        {
            vkWaitForFences(VulkanGlobal::context.device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
        }
        // Mark the image as now being in use by this frame
        imagesInFlight[imageIndex] = inFlightFences[currentFrame];

        updateScene(imageIndex);
        vkResetFences(VulkanGlobal::context.device, 1, &inFlightFences[currentFrame]);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore renderWaitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = renderWaitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[imageIndex];
        VkSemaphore renderSignalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = renderSignalSemaphores;

        if (vkQueueSubmit(VulkanGlobal::context.graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to submit draw command buffer!");
        }
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = renderSignalSemaphores;
        VkSwapchainKHR swapChains[] = {VulkanGlobal::swapchainContext.swapChain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr;

        result = vkQueuePresentKHR(VulkanGlobal::context.presentQueue, &presentInfo);

        if (result != VK_SUCCESS)
        {
            throw std::runtime_error("failed to present swap chain image!");
        }

        // Commented this out for playing around with it later :)
        // vkQueueWaitIdle(VulkanGlobal::context.presentQueue);
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    int nbFrames = 0;
    float lastTime = 0;
    void mainLoop()
    {
        while (!glfwWindowShouldClose(VulkanGlobal::context.window))
        {
            float currentTime = (float)glfwGetTime();
            deltaTime = currentTime - lastFrame;
            nbFrames++;
            if (currentTime - lastTime >= 1.0)
            { // If last prinf() was more than 1 sec ago
                // printf and reset timer
                printf("%f ms/frame\n", 1000.0 / double(nbFrames));
                nbFrames = 0;
                lastTime = currentTime;
            }
            lastFrame = currentTime;

            processInput(VulkanGlobal::context.window);
            glfwPollEvents();
            drawFrame();
        }

        vkDeviceWaitIdle(VulkanGlobal::context.device);
    }

    void initVulkan()
    {
        initScene();

        createCommandBuffers();
        createSyncObjects();
        glfwSetCursorPosCallback(VulkanGlobal::context.window, mouse_callback);
    }

    void cleanup()
    {
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vkDestroySemaphore(VulkanGlobal::context.device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(VulkanGlobal::context.device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(VulkanGlobal::context.device, inFlightFences[i], nullptr);
        }

        glfwTerminate();
    }
};

int main()
{
    HelloComputeApplication app;

    try
    {
        app.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        camera.ProcessKeyboard(UP, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        camera.ProcessKeyboard(DOWN, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
}

float lastX = 400, lastY = 300;
bool firstMouse = true;
void mouse_callback(GLFWwindow *window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}
