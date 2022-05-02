import matplotlib.pyplot as plt


def visualize(**images):
    """ plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20,8))

    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]); 
        plt.yticks([])
        
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)

    plt.show()