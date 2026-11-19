"""Shared image artist with an optional rendering backend."""


def show_image(app, axes, data, **kwargs):
    artist = axes.imshow(data, **kwargs)
    renderer = getattr(app, "image_renderer", None)
    if renderer is not None:
        renderer.attach(artist)
    return artist
