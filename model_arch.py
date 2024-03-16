import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import AutoModelForCausalLM
from PIL import Image

class CustomTransformer(nn.Module):
    def __init__(self, input_size, output_size, num_layers=2, hidden_size=256):
        super(CustomTransformer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=4, dim_feedforward=hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.linear(x)
        return x

class LargeWorldModel:
    def __init__(self, clip_model, clip_processor, transformer_config):
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.transformer = CustomTransformer(**transformer_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model.to(self.device)
        self.transformer.to(self.device)
        self.clip_model.eval()  # Set the CLIP model to evaluation mode
        self.transformer.eval()  # Set the transformer model to evaluation mode

    def encode_image(self, image_tensor, text=[' ']):
        # Provide a dummy text input along with the image
        inputs = self.clip_processor(text=text, images=image_tensor, return_tensors="pt")
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        return outputs.image_embeds

    def build_concept_space(self, image_path):
        #with torch.no_grad():
        # Preprocess the image
        image_tensor = self.preprocess_image(image_path)
        
        # Get embeddings for the image using the clip model
        embeddings = self.encode_image(image_tensor)

        # Save the embeddings to concept_space_1
        self.concept_space_1 = embeddings.cpu()  # Move embeddings to CPU for storage
        #print(f"len of cs1 {self.concept_space_1.shape}")

        # Pass embeddings through the custom transformer architecture
        concept_space_2 = self.transformer(embeddings)

        # Save the concept_space_2 embeddings to self variable
        self.concept_space_2 = concept_space_2.cpu()  # Move embeddings to CPU for storage
            #print(f"len of cs2 {self.concept_space_2.shape}")



    def preprocess_image(self, image_path):
        # Read image
        image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB mode
        
        # Preprocess image
        """
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize image to 224x224
            transforms.ToTensor(),           # Convert image to tensor
        ])
        image = preprocess(image)
        
        # Add batch dimension
        
        image = image.unsqueeze(0)
        """
        
        
        return image


class LWM_Actions():
    def __init__(self, clip_model, clip_processor, transformer_config, decoder_model="meta-llama/Llama-2-7b-hf"):
        #Define Encoders
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #define World Model 
        self.LWM = LargeWorldModel(clip_model, clip_processor, transformer_config)
    
        self.action_model = model = AutoModelForCausalLM.from_pretrained(decoder_model)
        self.action_model.eval()

    def build_action_space(self, image_path):
        self.LWM.build_concept_space(image_path)

        self.action_sequence = self.action_model(**self.LWM.concept_space_2, output_hidden_states=True).hidden_states[-1]


if __name__ == '__main__':
    #Imports are only here to avoid uncesseray over head during deployment 
    from transformers import CLIPProcessor, CLIPModel

    # Example usage:
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")  # Load your pre-trained CLIP model
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    transformer_config = {
        "input_size": 512,  # Assuming CLIP model's output size is 512
        "output_size": 512,  # Output size should match CLIP model's output size
        "num_layers": 2,     # Number of transformer layers
        "hidden_size": 256   # Hidden size of the transformer
    }

    concept_space_builder = LargeWorldModel(clip_model, clip_processor, transformer_config)
    image_path = "space_invaders_frames/frame_1.jpg"
    concept_space_builder.build_concept_space(image_path)

    # Now the concept_space_2 variable within concept_space_builder will hold the embeddings produced by the custom transformer architecture.
