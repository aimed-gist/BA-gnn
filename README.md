# Brain-Aware Readout Layers in GNNs: Advancing Alzheimer's early Detection and Neuroimaging

Alzheimer's disease (AD) is a neurodegenerative disorder characterized by progressive memory and cognitive decline, affecting millions worldwide. Diagnosing AD is challenging due to its heterogeneous nature and variable progression. This study introduces a novel brain-aware readout layer (BA readout layer) for Graph Neural Networks (GNNs), designed to improve interpretability and predictive accuracy in neuroimaging for early AD diagnosis. By clustering brain regions based on functional connectivity and node embedding, this layer improves the GNN's capability to capture complex brain network characteristics. We analyzed neuroimaging data from 383 participants, including both cognitively normal and preclinical AD individuals, using T1-weighted MRI, resting-state fMRI, and FBB-PET to construct brain graphs. Our results show that GNNs with the BA readout layer significantly outperform traditional models in predicting the Preclinical Alzheimer's Cognitive Composite (PACC) score, demonstrating higher robustness and stability. The adaptive BA readout layer also offers enhanced interpretability by highlighting task-specific brain regions critical to cognitive functions impacted by AD. These findings suggest that our approach provides a valuable tool for the early diagnosis and analysis of Alzheimer's disease.

## Citation
```bibtex
@inproceedings{HBAI_IJCAIW2024,
  title = {{Brain-Aware Readout Layers in GNNs: Advancing Alzheimer's early Detection and Neuroimaging}},
  author = {Jiwon Youn, Dong Woo Kang, Hyun Kook Lim, and Mansu Kim},
  booktitle = {IJCAI Workshop on Human Brain and Artificial Intelligence},
  year = {2024},
}
