[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11003180&assignment_repo_type=AssignmentRepo)
# SP23-CSE473-573-A-P3

SP23 CSE4/573-A Project 3

**CSE 473/573 Face Detection and Recognition Project.**

The following commands should be executed in the root folder of the project.

Please first update below and push your change to this repo.

- [your name]
- [ubit name]

**task 1 validation set**

```bash
# Face detection on validation data
python task1.py --input_path validation_folder/images --output ./result_task1_val.json

# Validation
python ComputeFBeta/ComputeFBeta.py --preds result_task1_val.json --groundtruth validation_folder/ground-truth.json
```

**task 1 test set running**

```bash
# Face detection on test data
python task1.py --input_path test_folder/images --output ./result_task1.json
```

**task 2 running**

```bash
python task2.py --input_path faceCluster_5 --num_cluster 5
```

**Pack your submission**
Note that when packing your submission, the script would run your code before packing.

```bash
sh pack_submission.sh <YourUBITName>
```

* Note: In the commands, use `python3` if your environment has python named as `python3` instead `python`.

Change **`<YourUBITName>`** with your UBIT name.
The resulting zip file should be named **"submission\_`<YourUBITName>`.zip"**, and it should contain 3 files, named **"result_task1.json"**, **"result_task2.json,"**, and **"UB\_Face.py"**. If not, there is something wrong with your code/filename, please go back and check.

You should only submit the zip file.
