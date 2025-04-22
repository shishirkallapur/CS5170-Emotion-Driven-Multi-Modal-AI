from audiocraft.models import MusicGen
from audiocraft.models import MultiBandDiffusion
import torchaudio
USE_DIFFUSION_DECODER = False


class AudioAgent:

    def __init__(self, modeltype: str):

        self.model = MusicGen.get_pretrained('facebook/musicgen-'+modeltype)
        self.model.set_generation_params(use_sampling = True,
                                         top_k = 250,
                                         duration = 30)
        
    def get(self, prompt):
        output = self.model.generate(descriptions = [prompt,],
                                     progress = True, return_tokens = True)

        # path = "C:\TouchDesigner projects"
        path = "D:\TouchDesigner projects"


        torchaudio.save(path+"/temp.wav", output[0][0].cpu(), 32000)
        print("Audio generated")