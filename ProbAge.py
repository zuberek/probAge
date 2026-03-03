import streamlit as st

st.info("""
### Probabilistic inference of epigenetic age acceleration from cellular dynamics  

Jan K. Dabrowski, Emma J. Yang, Samuel J. C. Crofts, Robert F. Hillary, Daniel J. Simpson, 
Daniel L. McCartney,  Riccardo E. Marioni, Kristina Kirschner, Eric Latorre-Crespo & Tamir Chandra  

*Nature Aging*, Volume 4, pages 1493–1507, published: 23 September 2024  

[Read the full paper](https://www.nature.com/articles/s43587-024-00700-5)
""")

st.markdown("""
## Welcome

This web application allows you to apply our published probabilistic epigenetic clock to your own DNA methylation dataset.  
You can run the full inference pipeline and explore participant-specific age acceleration and bias estimates directly in your browser.

### How to use the app

**1️⃣ Inference tab**  
Upload your methylation data and participant metadata, then run the inference using the provided button.  
The model will automatically adjust to your data and compute participant-level acceleration and bias estimates. Please wait until the process completes.

**2️⃣ Analysis tab**  
Explore the inferred results. You can zoom, select a participant, and inspect their posterior distribution of age acceleration and bias.

**3️⃣ CpG exploration tab**  
Investigate how CpG sites behave under the parameters of our site-level model and understand how site dynamics contribute to the clock.

---

For source code and full implementation details, visit our GitHub repository:  
https://github.com/zuberek/probAge
""")