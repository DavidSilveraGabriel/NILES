# NILES - Multimodal Chatbot with Gemini 🤖

NILES is an advanced multimodal chatbot leveraging Gemini models to process both text and images. Built with a modular architecture and a clean interface, NILES delivers an intuitive and powerful chat experience.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)  
![Streamlit](https://img.shields.io/badge/streamlit-1.31.0-red)  
![License](https://img.shields.io/badge/license-MIT-green)  

## 🌟 Features

- 💬 Intelligent text chat responses  
- 🖼️ Image analysis and processing  
- ⚙️ Flexible model parameter configuration  
- 🛡️ Robust error handling and validations  
- 📊 Image metadata visualization  
- 🔒 Secure implementation with input/output sanitization  

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/DavidSilveraGabriel/NILES.git
   cd niles
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```bash
   # Create a .env file in the project's root directory
   GOOGLE_API_KEY=your_google_api_key
   ```

## 📦 Project Structure

```
niles/
├── main.py           # Application entry point
├── config.py         # Configuration and constants
├── utils.py          # Utilities and helper functions
├── requirements.txt  # Project dependencies
├── .env              # Environment variables (not included in git)
└── README.md         # Documentation
```

## 💻 Usage

1. Run the application:
   ```bash
   streamlit run main.py
   ```

2. Access the web interface at `http://localhost:YOUR-PORT`.

3. Available features:
   - Text chat: Type messages and receive intelligent responses.
   - Image analysis: Upload images and ask questions about them.
   - Configuration: Adjust model parameters in the sidebar.

## ⚙️ Configuration

### Adjustable Parameters

- **Text Model**:
  - Temperature: Controls response creativity (0.0 - 1.0).
  - Maximum tokens: Response length limit (100 - 4096).

- **Vision Model**:
  - Top P: Controls response diversity.
  - Supported formats: JPG, PNG, WEBP.
  - Maximum file size: 5MB.

## 🔒 Security

- Input validation  
- Output sanitization  
- File size limits  
- Image format verification  
- Metadata cleanup  

## 🛠️ Development

### Prerequisites

- Python 3.8 or higher  
- Google Cloud account with Gemini API key  
- Streamlit  
- PIL (Pillow)  

### Core Dependencies

```python
streamlit==1.31.0
google-generativeai==0.3.1
python-dotenv==1.0.0
Pillow==10.2.0
```

## 📝 Model Configuration

The project utilizes two primary Gemini models:

1. **gemini-pro**: For text processing.  
2. **gemini-pro-vision**: For image analysis.  

## 🤝 Contribution

1. Fork the project.  
2. Create a new branch (`git checkout -b feature/amazing-feature`).  
3. Commit your changes (`git commit -m 'Add amazing feature'`).  
4. Push to the branch (`git push origin feature/amazing-feature`).  
5. Open a Pull Request.  

## 🐛 Reporting Issues

If you encounter any issues or have suggestions:

1. Check for similar issues first.  
2. Use the issue template to report problems.  
3. Include logs and screenshots if possible.  

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.  

## 📬 Contact

David Silvera - ingenieria.d.s.g@hotmail.com  

Project Link: [https://github.com/DavidSilveraGabriel/NILES](https://github.com/DavidSilveraGabriel/NILES)

## 🙏 Acknowledgments

- Google for the Gemini API.  
- Streamlit for the framework.  
- The Python community for the libraries used.  

---  
Made with ❤️ by [DavidSilveraGabriel](https://github.com/DavidSilveraGabriel)  and powered by 🤖[Claude](https://claude.ai/)

