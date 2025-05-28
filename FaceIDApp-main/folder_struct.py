import tempfile
import os

class folder_struct:
    def __init__(self):
        temp_dir = tempfile.gettempdir()

        if not os.path.exists(os.path.join(temp_dir,'data', 'verification_image')):
            os.makedirs(os.path.join(temp_dir,'data', 'verification_image'))
        if not os.path.exists(os.path.join(temp_dir,'data', 'input_image')):
            os.makedirs(os.path.join(temp_dir,'data', 'input_image'))
    
        self.VERIFICATION_IMAGE = os.path.join(temp_dir,'data', 'verification_image')
        self.INPUT_IMAGE = os.path.join(temp_dir,'data', 'input_image')