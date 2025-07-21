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

Please note that the weights of original model, retrained model with 3% forget size, retrained model with 6% forget size, and retrained model with 10% forget size are available here: [Weights](https://drive.google.com/drive/folders/15_3n6_fqDHVrgJLduWddzoT4wXwkGViQ?usp=sharing )

------- 

# Code Instructions

--------

# Citations and Resources

If you use this method or code, please cite:

```
@inproceedings{hardan2025forgetmi,
  title={Forget-MI: Machine Unlearning for Forgetting Multimodal Information in Healthcare Settings},
  author={Hardan, Shahad and Taratynova, Darya and Essofi, Abdelmajid and Nandakumar, Karthik and Yaqub, Mohammad},
  booktitle={Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2025}
}
```

Experiments were conducted on the MIMIC-CXR dataset:

Johnson, A.E.W., et al. (2019). MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports. Scientific Data, 6:317.
https://physionet.org/content/mimic-cxr/2.0.0/

A preprocessed subset of 6,742 image-report pairs from 1,663 subjects was used, focusing on multi-class edema severity classification, as adapted from [Chauhan et al., 2020].