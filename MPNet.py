from sentence_transformers import SentenceTransformer, models
import torch.nn as nn

#############load pretrained model######
"""
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
embeddings = model.encode(sentences)
print(embeddings)
"""

class vanilla_MPNet:
    def __init__(self, model_name = 'sentence-transformers/all-mpnet-base-v2', out_dim = 128):
        # load the model
        MPNet_model = models.Transformer(model_name)
        
        """
        self.classifier = nn.Sequential(
            nn.Linear(768,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim)
        )
        """
        """
        for param in MPNet_model.parameters():
            param.requires_grad = False
        """
        # set popling to collapse the sequence into a single fixed-size vector for classification
        pooling_model = models.Pooling(
            # pooling size, pooling type
            MPNet_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
        )
        
        # just select the nonlinear learnable and output size
        dense = models.Dense(
            # size out of pooling layer 768(default)
            in_features=pooling_model.get_sentence_embedding_dimension(),
            # output dim u want
            out_features=out_dim,
            # apply activation function
            #activation_function=nn.Tanh()
            activation_function=None
        )
        
        self.model = SentenceTransformer(modules=[MPNet_model, pooling_model, dense])
    
    def forward(self, feature):
        return self.model(feature)['sentence_embedding']
    