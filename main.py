import streamlit as st
from closest_country import find_closest_country
from format_image import format_image, im_to_polygon
import matplotlib.pyplot as plt
from shapely.affinity import scale, rotate, translate


st.title("Which country am I?")
st.write("Simply upload any image and find out which country is most similar.\n"
         "Built by [Dan Hirst](https://github.com/dchirst)")

im_stream = st.file_uploader("Upload an image you want to test:")

if im_stream is not None:
    # Remove background of image, to isolate main part of image
    im = format_image(im_stream)
    st.image(im)
    # Convert to a polygon to compare with the countries
    poly = im_to_polygon(im)

    # Perform country comparison algorithm
    pbar = st.progress(0)
    closest_country, cost, theta = find_closest_country(poly, pbar)

    # Plot the final country with image overlayed
    fig, ax = plt.subplots()
    scale_val, angle, xoff, yoff = tuple(theta)
    poly_transformed = scale(rotate(translate(poly, xoff=xoff, yoff=yoff), angle=angle), xfact=scale_val, yfact=scale_val)
    ax.plot(*poly_transformed.exterior.xy)
    plt.axis('off')
    closest_country.plot(ax=ax, color="#0fd159")
    st.title(f"The closest country is: {closest_country['ADMIN'].values[0]} ({cost})")

    st.pyplot(fig)

