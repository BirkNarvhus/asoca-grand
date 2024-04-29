import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from tqdm import tqdm
logits = np.load('logits.npy')
mask = np.load('masks.npy')
images = np.load('images.npy')


plt.figure(figsize=(20, 20))
fig, ax1 = plt.subplots()
#ax1.set_title('logits')
#ax2 = plt.subplot(1, 2, 2)
#ax2.set_title('mask')

img3d = np.moveaxis(np.array(images * 255, dtype=np.uint8), [-1, 2, 1], [1, -2, -1])
mask3d = np.moveaxis(np.array((1 - mask) * 255, dtype=np.uint8), [-1, 2, 1], [1, -2, -1])
logits3d = np.moveaxis(np.array(torch.sigmoid(torch.tensor(1 - logits)) * 255, dtype=np.uint8), [-1, 2, 1], [1, -2, -1])

im1 = ax1.imshow(np.concatenate([img3d[0][0], img3d[0][0]], axis=1), cmap="gray", animated=True)
ov1 = ax1.imshow(np.concatenate([logits3d[0][0], mask3d[0][0]], axis=1), cmap="jet", alpha=0.5, animated=True)

progress = tqdm(total=img3d.shape[0] * img3d.shape[1])



def update(i):
    img = np.concatenate([img3d[i // img3d.shape[1]][i % img3d.shape[1]], img3d[i // img3d.shape[1]][i % img3d.shape[1]]], axis=1)
    overlay = np.concatenate([logits3d[i // img3d.shape[1]][i % img3d.shape[1]], mask3d[i // img3d.shape[1]][i % img3d.shape[1]]], axis=1)
    im1.set_data(img)
    ov1.set_data(overlay)
    progress.update(1)
    return ov1,


animation_fig = animation.FuncAnimation(fig, update, frames=img3d.shape[0] * img3d.shape[1], interval=500, blit=True,repeat_delay=10,)
# Show the animation

animation_fig.save("overlayed_masks.gif")
progress.close()
