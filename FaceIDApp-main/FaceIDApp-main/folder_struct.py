import tempfile
import os

class folder_struct:
    def __init__(self):
        temp_dir = tempfile.gettempdir()

        if not os.path.exists(os.path.join(temp_dir,'data', 'verification_image')):
            os.makedirs(os.path.join(temp_dir,'data', 'verification_image'))
        if not os.path.exists(os.path.join(temp_dir,'data', 'input_image')):
            os.makedirs(os.path.join(temp_dir,'data', 'input_image'))
        if not os.path.exists(os.path.join(temp_dir,'data', 'save_model')):
            os.makedirs(os.path.join(temp_dir,'data', 'save_model'))
        if not os.path.exists(os.path.join(temp_dir,'data', 'threshold')):
            os.makedirs(os.path.join(temp_dir,'data', 'threshold'))
    
        self.VERIFICATION_IMAGE = os.path.join(temp_dir,'data', 'verification_image')
        self.INPUT_IMAGE = os.path.join(temp_dir,'data', 'input_image')
        self.SAVE_MODEL = os.path.join(temp_dir,'data', 'save_model')
        self.THRESHOLD = os.path.join(temp_dir,'data', 'threshold')