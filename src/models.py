import torch
import torch.nn as nn
from torchvision import models
import torch
import torch.nn as nn
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VivitImageProcessor, VivitForVideoClassification
import torch

class CustomCNN(nn.Module):
    def __init__(self, num_classes=101):
        super(CustomCNN, self).__init__()
                
        self.features = nn.Sequential(
            # 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),   
            nn.ReLU(),
            nn.MaxPool2d(2),      
            
            # 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # for optimizing the flatten dimension calculation, resolve bottleneck, reduce number of parameters drastically
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # after global pooling -> one for each feature map
        flatten_dim = 256
        
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),                 
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
     
        # dimensions of input tensor batch, channels, time, height, width
        # (16, 3, 16, 224, 224)
        b, c, t, h, w = x.size()
        
        # batch * frames, channels, height, width
        x = x.permute(0, 2, 1, 3, 4) 
        x = x.contiguous().view(b * t, c, h, w)

        # CNN
        x = self.features(x)      # Output: (B*T, 256, 14, 14)
        x = self.global_pool(x)   # Output: (B*T, 256, 1, 1)
        x = torch.flatten(x, 1)   # Output: (B*T, 256)

        # Temporal Aggregation
        # Average 16 vectors (one per frame) to get a single vector per video.
        # Return to shape (Batch, Frames, Features)
        x = x.view(b, t, -1)      # (B, 16, 256)
        x = x.mean(dim=1)         # Average over frames -> (B, 256)
        
        x = self.classifier(x)    # (B, 101)
        
        return x


#########################################################################################################


class ResNet2D_Baseline(nn.Module):
    def __init__(self, num_classes=101, pretrained=True, dropout_rate=0.5):
        super(ResNet2D_Baseline, self).__init__()
        
        # Load the backbone with pre-trained weights
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # FREEZING
        # We freeze the first layers (low-level features like edges/colors)
        for param in self.backbone.parameters():
            param.requires_grad = True # Enable all by default
            
        # Freeze up to layer2
        layers_to_freeze = [
            self.backbone.conv1, self.backbone.bn1, 
            self.backbone.layer1, self.backbone.layer2
        ]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

        # ROBUST HEAD
        # Replace the original FC layer with Dropout + Linear for better generalization
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        """
        Input: (Batch, Channels, Time, Height, Width) -> [B, 3, 16, 224, 224]
        """
        b, c, t, h, w = x.size()
        
        # Reshape to process each frame as an independent image
        # [B, 3, 16, 224, 224] -> [B, 16, 3, 224, 224] -> [B * 16, 3, 224, 224]
        x = x.transpose(1, 2).contiguous() 
        x = x.view(b * t, c, h, w)
        
        x = self.backbone(x) # Output: [B * 16, 101]
        
        # Restore temporal dimension
        # [B * 16, 101] -> [B, 16, 101]
        x = x.view(b, t, -1)
        
        # Temporal aggregation: Average over time dimension
        # [B, 16, 101] -> [B, 101]
        x = torch.mean(x, dim=1) 
        
        return x
    


#########################################################################################################

class CustomCNN_LSTM(nn.Module):
    def __init__(self, num_classes=101, lstm_hidden_size=256, lstm_layers=1, dropout_rate=0.5, checkpoint_path=None):
        super(CustomCNN_LSTM, self).__init__()
        
        base_model = CustomCNN(num_classes=num_classes)
        
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            base_model.load_state_dict(state_dict)
            print(f"Loaded pretrained CustomCNN weights from {checkpoint_path}")

        self.feature_extractor = nn.Sequential(
            base_model.features,
            base_model.global_pool
        )
        
        self.cnn_out_dim = 256
        
        self.lstm = nn.LSTM(input_size=self.cnn_out_dim, 
                            hidden_size=lstm_hidden_size, 
                            num_layers=lstm_layers, 
                            batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(lstm_hidden_size, num_classes)
        )

    def forward(self, x):
        # x shape: [Batch, Channels, Time, Height, Width] -> [B, 3, 16, 224, 224]
        b, c, t, h, w = x.size()
        
        # Step 1: Reshape CNN 
        # [B, 3, 16, 224, 224] -> [B*16, 3, 224, 224]
        x = x.transpose(1, 2).contiguous().view(b * t, c, h, w)
        
        # Feature Extraction CNN
        # Expected Output -> [B*16, 256, 1, 1]
        x = self.feature_extractor(x)
        
        # Step 3: Reshape LSTM
        # [B*16, 256] -> [B, 16, 256]
        x = x.view(b, t, -1)
        
        # out: [B, 16, lstm_hidden_size]
        # hn: [lstm_layers, B, lstm_hidden_size]
        out, (hn, cn) = self.lstm(x)
        
        # take last temportal step
        last_time_step = out[:, -1, :] # [B, lstm_hidden_size]
        
        # Final Logits
        x = self.classifier(last_time_step)
        
        return x
    

#########################################################################################################

class ResNet2D_LSTM(nn.Module):
    def __init__(self, num_classes=101, lstm_hidden_size=1024, dropout_rate=0.5):
        super(ResNet2D_LSTM, self).__init__()
        
        weights = models.ResNet18_Weights.DEFAULT
        resnet = models.resnet18(weights=weights)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        for param in self.feature_extractor.parameters():
            param.requires_grad = True
            
        self.spatial_dropout = nn.Dropout(p=dropout_rate)
            
        self.lstm = nn.LSTM(
            input_size=512, 
            hidden_size=lstm_hidden_size, 
            num_layers=2,           
            dropout=dropout_rate,  
            batch_first=True,
            bidirectional=False
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        b, c, t, h, w = x.size()
        
        # (B*T, C, H, W)
        x = x.transpose(1, 2).contiguous().view(b * t, c, h, w)
        
        # (B*T, 512, 1, 1) -> (B, T, 512)
        features = self.feature_extractor(x)
        features = features.view(b, t, -1)
        
        features = self.spatial_dropout(features)
        
        lstm_out, _ = self.lstm(features) # [B, T, 1024]
         
        pooled_out = torch.mean(lstm_out, dim=1) 
        
        return self.classifier(pooled_out)
    

#########################################################################################################
class ResNet2D_BiLSTM_Pro(nn.Module):
    def __init__(self, num_classes=101, lstm_hidden_size=512, dropout_rate=0.6):
        super(ResNet2D_BiLSTM_Pro, self).__init__()
        
        weights = models.ResNet18_Weights.DEFAULT
        resnet = models.resnet18(weights=weights)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        for param in self.feature_extractor.parameters():
            param.requires_grad = True
            
        self.spatial_dropout = nn.Dropout(p=dropout_rate)


        self.lstm = nn.LSTM(
            input_size=512, 
            hidden_size=lstm_hidden_size, 
            num_layers=2, 
            batch_first=True,
            bidirectional=True,
            dropout=0.5 if 2 > 1 else 0 
        )
        
        # lstm_hidden_size * 2 (BiLSTM)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x: [B, C, T, H, W]
        b, c, t, h, w = x.size()
        
        x = x.transpose(1, 2).contiguous().view(b * t, c, h, w)
        features = self.feature_extractor(x) # [B*T, 512, 1, 1]
        
        features = features.view(b, t, -1)
        
        features = self.spatial_dropout(features)
        
        # lstm_out shape: [B, T, hidden_size * 2]
        lstm_out, _ = self.lstm(features)
        
        # Temporal Pooling (Global Average Pooling on time dimension)
        pooled_out = torch.mean(lstm_out, dim=1) 
        
        return self.classifier(pooled_out)
    
##########################################################################################################

