import argparse
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground_truth_dir', default='./evaluation/quilt_1m_test_example_ground_truth.json')
    parser.add_argument('--pred_dir', default='./evaluation/quilt_1m_test_example_generated.json')
    args = parser.parse_args()

    annotation_file = args.ground_truth_dir
    results_file = args.pred_dir

    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    coco_eval = COCOEvalCap(coco, coco_result)

    coco_eval.params['image_id'] = coco_result.getImgIds()

    coco_eval.evaluate()


if __name__ == '__main__':
    main()
