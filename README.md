# Forget-MI: Machine Unlearning for Forgetting Multimodal Information in Healthcare Settings

## Summary about Forget-MI 

Forget-MI introduces a novel machine unlearning framework tailored to healthcare, where multimodal patient data (e.g., medical images and clinical text) must be removed upon request without compromising model utility. Unlike existing methods that target either unimodal data or are modality-agnostic, Forget-MI unlearns both unimodal and joint patient representations. This is achieved through a set of carefully designed loss functions that promote forgetting via noise-perturbed embeddings while retaining knowledge from non-forgotten data.

Forget-MI is evaluated on a subset of MIMIC-CXR for edema classification, demonstrating superior privacy protection (measured by Membership Inference Attacks) and reduced performance on the forget set, with negligible drops on the test set. The approach also avoids the inefficiencies of full retraining and scales across different forget-set sizes (3%, 6%, 10%).

The effectiveness of the method is illustrated in the Figure below, which outlines the multimodal unlearning process via four loss components: Unimodal Unlearning (UU), Multimodal Unlearning (MU), Unimodal Retention (UR), and Multimodal Retention (MR).

 <p align="center">
  <img src="forgetmi-Technical.png">
  </p>

The paper can be found here:[`arXiv preprint`](https://arxiv.org/abs/2506.23145v1) 

**Note:** This directory will be updated.

Please note that the weights of 1) original model, 2) retrained model with 3% forget size, 3) retrained model with 6% forget size, and 4) retrained model with 10% forget size are available here: [Weights](https://drive.google.com/drive/folders/15_3n6_fqDHVrgJLduWddzoT4wXwkGViQ?usp=sharing )

------- 

# Code Instructions