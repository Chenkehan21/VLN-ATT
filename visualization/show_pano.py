import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import MatterSim
import numpy as np


WIDTH = 50
HEIGHT = 640
VFOV = 95  #np.radians(95)
connectivity_dir = '../datasets/connectivity'
scan_dir = '../datasets/mp3d/v1/scans'


def build_simulator(connectivity_dir, scan_dir):
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setDatasetPath(scan_dir)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.initialize()
    return sim

sim = build_simulator(connectivity_dir, scan_dir)


def visualize_panorama_img(scan, viewpoint, heading, elevation):
    pano_img = np.zeros((HEIGHT, WIDTH*36, 3), np.uint8)
    for n_angle, angle in enumerate(range(-175, 180, 10)):
        sim.newEpisode([scan], [viewpoint], [heading + np.radians(angle)], [elevation])
        state = sim.getState()[0]
        im = np.array(state.rgb, copy=False)
        pano_img[:, WIDTH*n_angle:WIDTH*(n_angle+1), :] = im[..., ::-1]
    
    return pano_img


if __name__ == "__main__":
    pano_img = visualize_panorama_img("QUCTc6BB5sX", "f39ee7a3e4c04c6c8fd7b3f494d6504a", 3.665191429188092, 0)
    fig, ax = plt.subplots(1, 1, figsize=(25, 10))
    plt.tight_layout()
    ax.get_xaxis().set_ticks([],[])
    ax.get_yaxis().set_ticks([],[])
    ax.imshow(pano_img)
    plt.savefig("./pano.png")
    fig.clf()
    plt.close()
    print("done!")