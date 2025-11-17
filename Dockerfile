# Use stable Python
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "stock_dashboard_streamlit_pro_v5_5_6.py", "--server.port=8501", "--server.address=0.0.0.0"]
