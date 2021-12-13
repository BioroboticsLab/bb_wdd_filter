import matplotlib.pyplot as plt
import numpy as np
import sklearn.decomposition
import sklearn.manifold
import seaborn as sns
import torch
import tqdm.auto


def sample_embeddings(model, dataset, N=1000, test_batch_size=64, seed=42):
    n_batches = N // test_batch_size
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        random_samples = list(
            np.random.choice(
                len(dataset), size=n_batches * test_batch_size, replace=False
            )
        )
    finally:
        np.random.set_state(state)
    all_embeddings = []
    all_indices = []

    model.eval()
    with torch.no_grad():
        for _ in tqdm.auto.tqdm(range(n_batches), total=n_batches, leave=False):
            batch_images = []
            for _ in range(test_batch_size):
                idx = random_samples.pop()
                all_indices.append(idx)
                images, _ = dataset.__getitem__(idx, return_just_one=True)
                batch_images.append(images)
            batch_images = np.stack(batch_images, axis=0)
            batch_images = torch.from_numpy(batch_images).cuda()

            embeddings = model.embed(batch_images)
            all_embeddings.append(embeddings.detach().cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)[:, :, 0, 0]
    return embeddings, all_indices


def plot_embeddings(embeddings, indices, dataset, scatterplot=False, display=True):
    embeddings = sklearn.decomposition.PCA(10).fit_transform(embeddings)
    embeddings = sklearn.manifold.TSNE(
        2, learning_rate="auto", init="pca"
    ).fit_transform(embeddings)

    from PIL import Image

    min_x, max_x = embeddings[:, 0].min(), embeddings[:, 0].max()
    min_y, max_y = embeddings[:, 1].min(), embeddings[:, 1].max()

    W, H = 4000, 3000
    scale_x = W / (max_x - min_x)
    scale_y = H / (max_y - min_y)
    fig, ax = plt.subplots(figsize=(10, 10))
    if not scatterplot:
        embedding_image = Image.new("RGBA", (W, H))
        for (x, y), img_idx in zip(embeddings, indices):
            small = dataset.__getitem__(img_idx, return_just_one=True)[0][0]
            small = Image.fromarray((255.0 * (small + 1.0) / 2.0).astype(np.uint8))
            small.resize((64, 64))
            embedding_image.paste(
                small, (int((x - min_x) * scale_x), int((y - min_y) * scale_y))
            )
        ax.imshow(embedding_image)
    else:
        sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], alpha=0.5)

    ax.set_axis_off()

    if display:
        plt.show()
    else:
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return image
