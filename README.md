# Synthetic Image Generator using Vanilla GAN

This project implements a Vanilla Generative Adversarial Network (GAN) to generate
privacy-preserving synthetic images using the MNIST dataset.

## Motivation
In many domains such as healthcare, education, and surveillance, real image datasets
cannot be shared due to privacy regulations. This project demonstrates how GANs can
generate synthetic images that preserve data utility while protecting sensitive data.

## Project Modules
1. Data Pipeline & Preprocessing
2. Vanilla GAN Architecture
3. Training & Checkpointing
4. Evaluation & Quality Assurance
5. Streamlit Deployment
6. Monitoring & Future Improvements

## Tech Stack
- Python
- PyTorch
- Streamlit
- NumPy, Matplotlib
- Scikit-learn

## How to Run
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run src/app.py
Outputs
Synthetic MNIST digit images

Trained Generator & Discriminator

Evaluation metrics (Diversity, FID Proxy)

Interactive Streamlit web application

Use Cases
Healthcare, Retail, Education, Surveillance

Author

Team-8 GANS FOR IMAGES
