# dots_ocr_pipeline



## rotation fix 04.03.2026 (mert)
>> Tested rotation fix in the multimodal visualization script for:
Document: Preservative Efficacy Test - CofA_IND samples_Redacted
Page: 1, 3, 5
Problematic Page: 3 (already rotated bboxes/landscape bboxes)
Result: rotation_to_original method fixed the bbox-rotation problem. Bboxes were
drawn properly.
Meaning: Deepseek pipeline rotates pages and OCRs them in landscape rotation. 
This creates bbox coordinates for landscape page orientation. We need to revert 
that to original page orientation for FE visualizations