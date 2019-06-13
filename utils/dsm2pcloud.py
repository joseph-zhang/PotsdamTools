import time
import argparse
import tifffile
import numpy as np
from open3d import read_point_cloud, draw_geometries


parser = argparse.ArgumentParser(description='Convert DSM to Point Cloud')
parser.add_argument('--rgb_path',      type=str,     required=True,             help='The path of input RGB TIFF file')
parser.add_argument('--dsm_path',      type=str,     required=True,             help='The path of input DSM TIFF file')
parser.add_argument('--out_path',      type=str,     default="./pcloud.ply",    help='The path of output ply file')
parser.add_argument('--scale_factor',  type=float,   default=1.0,               help='the factor * input value')
args = parser.parse_args()


class DSM2Cloud():
    def __init__(self,
                 rgb_file,
                 dsm_file,
                 out_file,
                 scalingfactor):
        self.rgb_file = rgb_file
        self.dsm_file = dsm_file
        self.out_file = out_file
        self.scalingfactor = scalingfactor
        self.rgb_image = tifffile.imread(rgb_file).astype(np.uint8)
        self.dsm_image = tifffile.imread(dsm_file).astype(np.float32)
        self.img_shape = self.rgb_image.shape
        self.img_height = self.img_shape[0]
        self.img_width = self.img_shape[1]

    def gen_cloud(self):
        Z = self.dsm_image / self.scalingfactor
        X = np.zeros([self.img_height, self.img_width])
        Y = np.zeros([self.img_height, self.img_width])
        for i in range(self.img_height):
            X[i,:] = np.full(self.img_width, i)

        for i in range(self.img_width):
            Y[:,i] = np.full(self.img_height, i)

        points_rgb = np.zeros([6, self.img_height*self.img_width])
        points_rgb[0] = (X - self.img_height / 2).reshape(-1)
        points_rgb[1] = (Y - self.img_width / 2).reshape(-1)
        points_rgb[2] = Z.reshape(-1)
        points_rgb[3] = self.rgb_image[:,:,0].reshape(-1)
        points_rgb[4] = self.rgb_image[:,:,1].reshape(-1)
        points_rgb[5] = self.rgb_image[:,:,2].reshape(-1)
        self.points_rgb = points_rgb

    def write_cloudfile(self):
        points = []
        float_formatter = lambda x: "%.4f" % x
        for point in self.points_rgb.T:
            points.append("{} {} {} {} {} {} 0\n".format(float_formatter(point[0]),
                                                         float_formatter(point[1]),
                                                         float_formatter(point[2]),
                                                         int(point[3]),
                                                         int(point[4]),
                                                         int(point[5])))
        with open(self.out_file, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('element vertex %d\n'%(len(points)))
            f.write('property float32 x\n')
            f.write('property float32 y\n')
            f.write('property float32 z\n')
            f.write('property uchar red\n')
            f.write('property uchar green\n')
            f.write('property uchar blue\n')
            f.write('property uchar alpha\n')
            f.write('end_header\n')
            f.write('%s'%("".join(points)))

    def show_point_cloud(self):
        pcd = read_point_cloud(self.out_file)
        draw_geometries([pcd])


if __name__ == '__main__':
    genner = DSM2Cloud(args.rgb_path, args.dsm_path, args.out_path, scalingfactor=args.scale_factor)
    genner.gen_cloud()
    genner.write_cloudfile()
    genner.show_point_cloud()
