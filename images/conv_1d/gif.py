import imageio
import os 

images = []
for f in range(0, 7):
	images.append(imageio.imread(str(f) + ".jpg"))

print(images)

imageio.mimsave("gif.gif", images, duration=1)
