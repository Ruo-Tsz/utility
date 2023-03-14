import glob
from PIL import Image, ImageDraw
import os

# filepaths
# fp_in = "/path/to/image_*.png"
fp_in = '/data/progress/2022/gt_compare/frame_num_2/map_v3/270_16615/prolong_all_1.0s/test/*.png'
fp_out = "/data/progress/2022/gt_compare/frame_num_2/map_v3/270_16615/prolong_all_1.0s/gif"

imgs = (Image.open(f) for f in sorted(glob.glob(fp_in)))
img = next(imgs)  # extract first image from iterator
img.save(fp=fp_out + "/test.gif", format='GIF', append_images=imgs,
         save_all=True, duration=200, loop=0)


# Create the frames
# frames = []

# for frame in sorted(os.listdir(fp_in)):
#     new_frame = Image.open(fp_in + "/" + frame)
#     frames.append(new_frame)
#     print(frame)
# print(len(frames))
# # Save into a GIF file
# frames[0].save(fp_out + "/animation.gif", format='GIF',
#                append_images=frames[1:],
#                save_all=True,
#                duration=500, loop=0)


# import imageio
# images = []
# for filename in fp_in:
#     images.append(imageio.imread(filename))
# imageio.mimsave(fp_out, images)

# import imageio
# with imageio.get_writer('/path/to/movie.gif', mode='I') as writer:
#     for filename in filenames:
#         image = imageio.imread(filename)
#         writer.append_data(image)