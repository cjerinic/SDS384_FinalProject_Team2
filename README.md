# SDS384 Team 2 Final Project
Project title: Classifying Complex Cognitive Operations from fMRI Data

Brief description: We used various ML classification algorithms with labeled fMRI data to decode mental operations during a working memory experiment.

Data description:
- 5 subjects
- BOLD data masked with whole-brain mask (we looked at voxels across entire brain)
- (1098, 151424) [time x features] for each subject
- Feature selection to reduce the dimensionality of the data (1098, 19347)
- 3 unique cognitive operations labeled
