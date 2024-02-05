# SAMCell-GUI

This is the GUI associated with the SAMCell paper: "SAMCell: Generalized Label-Free Biological Cell Segmentation with Segment Anything".  We recommend using this GUI only if you have a sufficiently powerful computer (a recent high-end Nvidia GPU).  Otherwise, we recommend our [Google Colab Notebook instead](https://colab.research.google.com/drive/1016jr1JTtSI4kUIaHmnmXn290n-SM2eP).

## Installation

To use this GUI you need to install Python 3.11. You can download it from [here](https://www.python.org/downloads/).  Python works with all major operating systems, so you can use this GUI on Windows, MacOS, and Linux.

Next, navigate to the directory where you have downloaded the GUI and run the following commands in terminal:
    
    cd path/to/SAMCell-GUI
    
    pip install -r requirements.txt

Finally, download the pretrained SAMCell model from the base SAMCell repo [here](https://github.com/NathanMalta/SAMCell/releases/download/v1/samcell-cyto.zip).  Unzip the model and place it in the `SAMCell-GUI` directory.  Your file structure should look like this:

    SAMCell-GUI
    ├── README.md
    ├── requirements.txt
    ...
    ├── samcell-cyto
    │   ├── ... (pretained model files)

## Usage

To start the GUI, run the following command in terminal (while cd'd into the `SAMCell-GUI` directory):
    
    python gui.py

A window like the one below should appear:
![gui](https://github.com/NathanMalta/SAMCell-GUI/blob/main/media/gui-empty.png?raw=true)

Click and drag a few files into the window.  They should appear in the list on the left.  Then click "Process this Image" to run SAMCell on the selected image.  After processing is complete, the file name will turn green and you can view the results in the "Segmentation Results" and "Metrics" tabs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.