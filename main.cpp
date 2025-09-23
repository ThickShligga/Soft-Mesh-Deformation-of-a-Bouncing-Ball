#include <igl/read_triangle_mesh.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/edges.h>
#include <igl/unproject.h>
#include <Eigen/Dense>
#include <iostream>
#include <vector>

struct Particle {
    Eigen::Vector3d pos, vel, prev_pos;
    double mass;
};

struct Spring {
    int i, j;
    double L0;
    double* k;
    double* damping;
};

int main(int argc, char* argv[]) {
    
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    std::string input_file = (argc > 1) ? argv[1] : "../../../IcoSphere.obj";
    if (!igl::read_triangle_mesh(input_file, V, F)) {
        std::cerr << "Failed to load " << input_file << std::endl;
        return 1;
    }
    //Tetrahedralize
    Eigen::MatrixXd TV;
    Eigen::MatrixXi TT, TF;
    if (igl::copyleft::tetgen::tetrahedralize(V, F, "pq1.414Y", TV, TT, TF) != 0) {
        std::cerr << "Tetrahedralization failed" << std::endl;
        return 1;
    }

    //std::cout << "Vertices: " << TV.rows() << ", Tetrahedra: " << TT.rows() << std::endl;

    //Initialize particles
    std::vector<Particle> initial_particles; //for resetting
    std::vector<Particle> particles(TV.rows());
    for (int i = 0; i < TV.rows(); ++i) {
        particles[i].pos = TV.row(i).transpose();
        particles[i].prev_pos = particles[i].pos;
        particles[i].vel = Eigen::Vector3d::Zero();
        particles[i].mass = 0.1;
        
    }

    initial_particles = particles;

    
    Eigen::MatrixXi E;
    igl::edges(TT, E);
    std::vector<Spring> springs(E.rows());

    double globalK = 26000;
    double globalD = 14.0;

    //Initialize Springs
    for (int i = 0; i < E.rows(); ++i) {
        springs[i].i = E(i, 0);
        springs[i].j = E(i, 1);

        if (springs[i].i >= TV.rows() || springs[i].j >= TV.rows()) {
            std::cerr << "Invalid spring indices: " << springs[i].i << ", " << springs[i].j << std::endl;
            continue;
        }

        Eigen::Vector3d p1 = particles[springs[i].i].pos;
        Eigen::Vector3d p2 = particles[springs[i].j].pos;
        springs[i].L0 = (p2 - p1).norm();
        springs[i].k = &globalK;
        springs[i].damping = &globalD;
    }

    //Simulation parameters for use with verlet integration
    double dt = 0.01; //use 0.01
    int substeps = 24; //use 12 for most ball like
    double subdt = dt / substeps;
    Eigen::Vector3d gravity(0, -9.81, 0);
    double ground_y = -5.0;

    
    bool is_dragging = false;
    int dragged_vertex = -1;
    Eigen::Vector3d drag_target;

    
    igl::opengl::glfw::Viewer viewer;
    viewer.core().light_position = Eigen::Vector3f(0.0, 10.0, 0.0);
    viewer.core().lighting_factor = 0.0;
    

    //Create new mesh to be updated with particle positions
    Eigen::MatrixXd current_TV = TV;
    for (int i = 0; i < particles.size(); ++i) {
        current_TV.row(i) = particles[i].pos.transpose();
    }

    viewer.data().set_mesh(current_TV, TF);
    viewer.data().point_size = 10;

    //Ground plane for visualization
    viewer.append_mesh();
    Eigen::MatrixXd ground_V(4, 3);
    ground_V << -10, ground_y, -10,
        -10, ground_y, 10,
        10, ground_y, 10,
        10, ground_y, -10;

    Eigen::MatrixXi ground_F(2, 3);
    ground_F << 0, 1, 2,
        0, 2, 3;

    viewer.data(1).set_mesh(ground_V, ground_F);
    viewer.data(1).set_colors(Eigen::RowVector3d(1, 1, 1));
    viewer.data(0).set_colors(Eigen::RowVector3d(1, 0, 0));

    

    //Select nearest vertex to drag
    viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& viewer, int button, int) -> bool {
        if (button == 0) { // Left mouse button
            float mouse_x = static_cast<float>(viewer.current_mouse_x);
            float mouse_y = viewer.core().viewport(3) - static_cast<float>(viewer.current_mouse_y);

            Eigen::Vector3f near_pos(mouse_x, mouse_y, 0.0f);
            Eigen::Vector3f far_pos(mouse_x, mouse_y, 1.0f);
            Eigen::Vector3f ray_origin, ray_dir_far;
            igl::unproject(near_pos, viewer.core().view, viewer.core().proj, viewer.core().viewport, ray_origin);
            igl::unproject(far_pos, viewer.core().view, viewer.core().proj, viewer.core().viewport, ray_dir_far);
            Eigen::Vector3d ray_o = ray_origin.cast<double>();
            Eigen::Vector3d ray_d = (ray_dir_far - ray_origin).normalized().cast<double>();

            //Find nearest vertex to the ray
            double min_dist = std::numeric_limits<double>::max();
            dragged_vertex = -1;

            for (int i = 0; i < particles.size(); ++i) {
                Eigen::Vector3d oc = ray_o - particles[i].pos;
                double b = 2.0 * oc.dot(ray_d);
                double c = oc.dot(oc);
                double t = -b / (2.0 * ray_d.dot(ray_d));
                if (t >= 0) {
                    Eigen::Vector3d closest_point = ray_o + ray_d * t;
                    double dist = (closest_point - particles[i].pos).norm();
                    if (dist < 0.5 && dist < min_dist) {
                        min_dist = dist;
                        dragged_vertex = i;
                        drag_target = particles[i].pos;
                    }
                }
            }

            if (dragged_vertex != -1) {
                is_dragging = true;
                //std::cout << "Dragging vertex: " << dragged_vertex << " at " << drag_target.transpose() << std::endl;
                return true;
            }
        }
        return false;
    };

    //Update drag target position
    viewer.callback_mouse_move = [&](igl::opengl::glfw::Viewer& viewer, int x, int y) -> bool {
        if (is_dragging) {
            float mouse_x = static_cast<float>(x);
            float mouse_y = viewer.core().viewport(3) - static_cast<float>(y);

            Eigen::Vector3f near_pos(mouse_x, mouse_y, 0.0f);
            Eigen::Vector3f far_pos(mouse_x, mouse_y, 1.0f);
            Eigen::Vector3f ray_origin, ray_dir_far;
            igl::unproject(near_pos, viewer.core().view, viewer.core().proj, viewer.core().viewport, ray_origin);
            igl::unproject(far_pos, viewer.core().view, viewer.core().proj, viewer.core().viewport, ray_dir_far);
            Eigen::Vector3d ray_o = ray_origin.cast<double>();
            Eigen::Vector3d ray_d = (ray_dir_far - ray_origin).normalized().cast<double>();

            double t = (drag_target - ray_o).dot(ray_d) / ray_d.dot(ray_d);
            drag_target = ray_o + ray_d * t;
            return true;
        }
        return false;
    };

    //Stop dragging on mouse up
    viewer.callback_mouse_up = [&](igl::opengl::glfw::Viewer&, int button, int) -> bool {
        if (button == 0 && is_dragging) {
            is_dragging = false;
            dragged_vertex = -1;
            //std::cout << "Dragging stopped" << std::endl;
            return true;
        }
        return false;
    };

    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer&) -> bool {
        

        for (int step = 0; step < substeps; ++step) {
            
            std::vector<Eigen::Vector3d> forces(particles.size(), Eigen::Vector3d(0,-0.981,0));

            /*for (int i = 0; i < particles.size(); ++i) {
                forces[i] = gravity * particles[i].mass;
            }*/

            for (const auto& s : springs) {
                if (s.i >= particles.size() || s.j >= particles.size()) {
                    continue;
                }

                Eigen::Vector3d p1 = particles[s.i].pos;
                Eigen::Vector3d p2 = particles[s.j].pos;
                Eigen::Vector3d dp = p2 - p1;
                double L = dp.norm();

                if (L < 1e-1) continue;

                Eigen::Vector3d normalizedDirection = dp / L;
                double deltaX = L - s.L0;

                Eigen::Vector3d F_s = *s.k * deltaX * normalizedDirection;
                Eigen::Vector3d relativeVelocity = particles[s.j].vel - particles[s.i].vel;
                Eigen::Vector3d F_d = *s.damping * relativeVelocity.dot(normalizedDirection) * normalizedDirection;

                 forces[s.i] += F_s + F_d;
                 forces[s.j] -= F_s + F_d;
            }

            //Dragging force
            if (is_dragging && dragged_vertex != -1) {
                Eigen::Vector3d drag_force = (drag_target - particles[dragged_vertex].pos) * 500.0;
                forces[dragged_vertex] += drag_force;
            }

            //Verlet integration for actual simulation
            for (int i = 0; i < particles.size(); ++i) {
                
                Eigen::Vector3d accel = forces[i] / particles[i].mass;
                Eigen::Vector3d vel = (particles[i].pos - particles[i].prev_pos) / subdt;
                //vel *= 0.99;

                Eigen::Vector3d new_pos = particles[i].pos + vel * subdt + 0.5*accel * subdt * subdt;

                particles[i].vel = (new_pos - particles[i].pos) / subdt;
                particles[i].prev_pos = particles[i].pos;
                particles[i].pos = new_pos;

                if (particles[i].pos.y() < ground_y) {
                    particles[i].pos.y() = ground_y;
                    particles[i].vel.y() *= -0.20;
                    particles[i].vel.x() *= 0.95;
                    particles[i].vel.z() *= 0.95;
                    particles[i].prev_pos = particles[i].pos - particles[i].vel * subdt;
                }
            }
        }

        //Update visualization
        for (int i = 0; i < particles.size(); ++i) {
            current_TV.row(i) = particles[i].pos.transpose();
        }

        viewer.data(0).set_vertices(current_TV);
        viewer.data(0).compute_normals();

        return false;
    };


    std::cout << "Press r/R to reset" << std::endl;
    viewer.callback_key_down = [&](igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier) -> bool {
        if (key == 'R' || key == 'r') {
            particles = initial_particles;
            for (int i = 0; i < particles.size(); ++i) {
                current_TV.row(i) = particles[i].pos.transpose();
            }
            globalK = 24000;
            globalD = 14.0;
            substeps = 12;
            viewer.data(0).set_vertices(current_TV);
            viewer.data(0).compute_normals();
            std::cout << "Simulation reset to initial state" << std::endl;
            return true;
        }
        else if (key == 'U') {
            substeps += 1;
            std::cout << "substeps increased to: " << substeps << std::endl;
            return true;
        }
        else if (key == 'J') {
            substeps = std::max(1, substeps - 1);
            std::cout << "substeps decreased to: " << substeps << std::endl;
            return true;
        }
        else if (key == 'K' && (modifier & GLFW_MOD_SHIFT)) {
            globalK += 500.0;
            for (auto& s : springs) *s.k = globalK;
            std::cout << "Spring stiffness increased to: " << globalK << std::endl;
            return true;
        }
        else if (key == 'K' && !(modifier & GLFW_MOD_SHIFT)) {
            globalK = std::max(500.0, globalK - 500.0);
            for (auto& s : springs) *s.k = globalK;
            std::cout << "Spring stiffness decreased to: " << globalK << std::endl;
            return true;
        }
        else if (key == 'D' && (modifier & GLFW_MOD_SHIFT)) {
            globalD += 0.5;
            for (auto& s : springs) *s.damping = globalD;
            std::cout << "Damping increased to: " << globalD << std::endl;
            return true;
        }
        else if (key == 'D' && !(modifier & GLFW_MOD_SHIFT)) {
            globalD = std::max(0.0, globalD - 0.5);
            for (auto& s : springs) *s.damping = globalD;
            std::cout << "Damping decreased to: " << globalD << std::endl;
            return true;
        }
        return false;
    };


    viewer.data(0).compute_normals();
    viewer.core().animation_max_fps = 60.0;
    viewer.core().is_animating = true;
    viewer.launch();

    return 0;
}