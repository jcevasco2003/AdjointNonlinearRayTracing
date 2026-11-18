
#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <Renderer/Shader.h>
#include <Renderer/Camera.h>

// GLM: vectores, matrices y transformaciones
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <memory>

// ============================================================
// CONFIGURACION DE LA VENTANA
// ============================================================

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// ============================================================
// CAMARA DEL OBSERVADOR
// ============================================================

// Esta es la camara desde la cual observamos la escena.
// NO es la fuente de rayos opticos.
//
// Los movimientos WASD y el mouse solamente modifican
// nuestra perspectiva de visualizacion.

Camera viewerCamera(glm::vec3(0.0f, 0.0f, 5.0f));

// Control del tiempo entre frames
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// Control del mouse
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// ============================================================
// CALLBACKS DE GLFW
// ============================================================

// Se ejecuta cuando cambia el tamanio de la ventana.
void framebuffer_size_callback(
    GLFWwindow* window,
    int width,
    int height)
{
    glViewport(0, 0, width, height);
}

// ============================================================
// INPUT DE TECLADO
// ============================================================

void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    // Movimiento de la camara del observador
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        viewerCamera.ProcessKeyboard(FORWARD, deltaTime);

    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        viewerCamera.ProcessKeyboard(BACKWARD, deltaTime);

    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        viewerCamera.ProcessKeyboard(LEFT, deltaTime);

    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        viewerCamera.ProcessKeyboard(RIGHT, deltaTime);
}

// ============================================================
// INPUT DEL MOUSE
// ============================================================

// Rotacion de la camara del observador.
void mouse_callback(
    GLFWwindow* window,
    double xposIn,
    double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;

    lastX = xpos;
    lastY = ypos;

    viewerCamera.ProcessMouseMovement(xoffset, yoffset);
}

// Zoom de la camara del observador.
void scroll_callback(
    GLFWwindow* window,
    double xoffset,
    double yoffset)
{
    viewerCamera.ProcessMouseScroll(
        static_cast<float>(yoffset)
    );
}

// ============================================================
// INICIALIZACION DE LA VENTANA
// ============================================================

GLFWwindow* initWindow()
{
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW\n";
        return nullptr;
    }

    // Solicitamos OpenGL 3.3 Core.
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(
        SCR_WIDTH,
        SCR_HEIGHT,
        "Nonlinear Optical Ray Tracer",
        nullptr,
        nullptr
    );

    if (!window)
    {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return nullptr;
    }

    glfwMakeContextCurrent(window);

    glfwSetFramebufferSizeCallback(
        window,
        framebuffer_size_callback
    );

    // GLAD 2: cargamos las funciones de OpenGL
    if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress))
    {
        std::cerr << "Failed to initialize GLAD\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return nullptr;
    }

    // Inicializamos el viewport con las dimensiones actuales.
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);

    return window;
}

// ============================================================
// MAIN
// ============================================================

int main()
{
    // --------------------------------------------------------
    // 1. INICIALIZAR VENTANA Y OPENGL
    // --------------------------------------------------------

    GLFWwindow* window = initWindow();

    if (!window)
        return -1;

    // Ocultamos el cursor para controlar la camara con el mouse.
    glfwSetInputMode(
        window,
        GLFW_CURSOR,
        GLFW_CURSOR_DISABLED
    );

    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    glfwSwapInterval(1);

    // Habilitamos el test de profundidad.
    glEnable(GL_DEPTH_TEST);

    // --------------------------------------------------------
    // 2. SHADERS
    // --------------------------------------------------------

    // TODO: todavia tenes que crear estos archivos GLSL.
    //
    // basic.vert:
    //   Transformaciones de los vertices.
    //
    // basic.frag:
    //   Color de los objetos de la escena.
    //
    // Por ahora no instanciamos shaders para rayos ni volumenes.

    // Shader basicShader(
    //     "shaders/basic/basic.vert",
    //     "shaders/basic/basic.frag"
    // );

    // --------------------------------------------------------
    // 3. ESCENA OPTICA
    // --------------------------------------------------------

    // TODO: implementar Scene.
    //
    // Scene scene;
    //
    // La escena debera contener:
    //   - Medio optico.
    //   - Emisor de rayos.
    //   - Sensores, si corresponden.
    //
    // Importante: el emisor optico es independiente de
    // viewerCamera.

    // --------------------------------------------------------
    // 4. MEDIO NO LINEAL
    // --------------------------------------------------------

    // TODO: implementar Medium y LuneburgMedium.
    //
    // Ejemplo conceptual:
    //
    // auto medium = std::make_shared<LuneburgMedium>();
    // scene.addMedium(medium);
    //
    // El medio definira el indice de refraccion:
    //
    // n(x, y, z)
    //
    // Inicialmente podemos usar una funcion analitica.
    // Mas adelante podremos reemplazarla por un campo voxelizado.

    // --------------------------------------------------------
    // 5. SIMULACION DE RAYOS
    // --------------------------------------------------------

    // TODO: implementar Ray, RayEmitter y Tracer.
    //
    // Tracer tracer;
    //
    // tracer.trace(scene);
    //
    // El resultado sera una coleccion de trayectorias:
    //
    // vector<RayTrajectory>
    //
    // Cada trayectoria contendra una secuencia de posiciones
    // tridimensionales calculadas por el simulador.

    // --------------------------------------------------------
    // 6. RENDERER
    // --------------------------------------------------------

    // TODO: implementar Renderer.
    //
    // Renderer renderer;
    //
    // Su responsabilidad sera dibujar:
    //   - La geometria del medio.
    //   - Las trayectorias de los rayos.
    //   - Emisores y sensores.
    //
    // El renderer no calcula la fisica de propagacion.

    // --------------------------------------------------------
    // 7. LOOP PRINCIPAL
    // --------------------------------------------------------

    while (!glfwWindowShouldClose(window))
    {
        // ----------------------------------------------------
        // Actualizacion del tiempo
        // ----------------------------------------------------

        float currentFrame =
            static_cast<float>(glfwGetTime());

        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // ----------------------------------------------------
        // Input
        // ----------------------------------------------------

        processInput(window);

        // ----------------------------------------------------
        // Limpiar buffers
        // ----------------------------------------------------

        glClearColor(
            0.08f,
            0.09f,
            0.12f,
            1.0f
        );

        glClear(
            GL_COLOR_BUFFER_BIT |
            GL_DEPTH_BUFFER_BIT
        );

        // ----------------------------------------------------
        // Camara del observador
        // ----------------------------------------------------

        // Matriz de proyeccion:
        // perspectiva de la escena 3D.

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        if (height > 0)
        {
            glm::mat4 projection = glm::perspective(
                glm::radians(viewerCamera.Zoom),
                static_cast<float>(width) / height,
                0.1f,
                100.0f
            );

            // Matriz de vista:
            // posicion y orientacion del observador.

            glm::mat4 view = viewerCamera.GetViewMatrix();

            // ------------------------------------------------
            // Dibujado de la escena
            // ------------------------------------------------

            // TODO: dibujar el medio optico.
            //
            // renderer.drawMedium(scene, view, projection);

            // TODO: dibujar los rayos calculados.
            //
            // renderer.drawRays(
            //     tracer.getTrajectories(),
            //     view,
            //     projection
            // );

            // TODO: dibujar emisores y sensores.
        }

        // ----------------------------------------------------
        // Presentar frame
        // ----------------------------------------------------

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // --------------------------------------------------------
    // FINALIZACION
    // --------------------------------------------------------

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
