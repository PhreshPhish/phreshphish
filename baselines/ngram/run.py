import argparse
from train import train_model, test_model
from feat_processing import create_libsvm_feat, reduce_feat_file


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train and Test Feed Forward DNN model for Phishing Benchmark"
    )
    parser.add_argument(
        "--action",
        type=str,
        required=True,
        help="Train or Test.",
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="The path to the trian/test dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="The path to the output files and models.",
    )

    args = parser.parse_args()

    if args.action == "train":
        
        create_libsvm_feat(input_path=args.input_path, 
                           feature_dict_file=f"{args.output_path}/feat_dict.txt", 
                           label_dict_file=f"{args.output_path}/label_dict.txt", 
                           feat_file=f"{args.output_path}/train_feat_file.libsvm", 
                           train_mode=True)

        reduce_feat_file(input_feat=f"{args.output_path}/train_feat_file.libsvm", 
                        input_dict=f"{args.output_path}/feat_dict.txt", 
                        output_feat=f"{args.output_path}/train_feat_file_reduced.libsvm", 
                        output_dict=f"{args.output_path}/feat_dict_reduced.txt",
                        max_feats=300000)    
                
        train_model(train_file=f"{args.output_path}/train_feat_file_reduced.libsvm", 
                    dict_file=f"{args.output_path}/feat_dict_reduced.txt",
                    model_path=f"{args.output_path}/model", batch_size=8, epochs=1)

    elif args.action == "test":
        
        create_libsvm_feat(input_path=args.input_path, 
                           feature_dict_file=f"{args.output_path}/feat_dict_reduced.txt", 
                           label_dict_file=f"{args.output_path}/label_dict.txt", 
                           feat_file=f"{args.output_path}/test_feat_file.libsvm", 
                           train_mode=False)
        
        test_model(test_file=f"{args.output_path}/test_feat_file.libsvm", 
                   dict_file=f"{args.output_path}/feat_dict_reduced.txt", 
                   label_dict_file=f"{args.output_path}/label_dict.txt", 
                   model_path=f"{args.output_path}/model", 
                   report_file= f"{args.output_path}/test_report.txt",
                   batch_size=8)
    else:
        print(f"Invalid action: {args.action}")