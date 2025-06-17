# Center Prediction for Retrosynthesis using GNNs with Attention

## Abstract
Retrosynthesis is one of the most important techniques in Organic Chemistry, offering a "backward" approach in Organic Synthesis, with prevalent applications in the field of Drug Discovery and Drug Development. While the final goal of Organic Synthesis is Synthesis itself, Retrosynthesis is crucial in finding highly available and / or high quality pathways to achieve it, deconstructing known product molecules into sets of possible precursors. Historically an extremely complex and time-intensive task that is heavily reliant on prior domain knowledge and expertise (known reactions and compounds in the 10<sup>7</sup>s), Deep Learning can be used to immensely benefit it by "quickly" offering pathway predictions to be later reviewed by the actual experts, vastly accelerating the process. In this context and that of Deep Learning, our Assignment aims to offer a trivial implementation of the first steps of Retrosynthesis pipeline, building a model for Chemical Reaction Center prediction i.e. for finding which of the product's bonds can be broken, with the aim of splitting it into valid predecessors.


## Dataset

Our model is trained and tested on USPTO-50k (United States Patent and Trademark Office), a widely recognized dataset in ML assisted Organic Synthesis that consists of 50K extracted atom-mapped reactions in the form of SMILES (Simplified Molecular Input Line Entry System) strings.

## Model Used

**TransformerConv**

One of torch_geometric's basic Transformer - GNN models offering abstraction for the Message Passing process