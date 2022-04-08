## Task modality in fMRI code

This repo contains the source code for the calculations done in the forthcoming paper *Task modality in fMRI is detected by persistence*.

The structure is as follows:
 - `data` contains three subdirectories for getting the fMRI data into the correct format, sorted further by subjectID.
   - `raw` contains the raw matlab files.
   - `preprocessed` contains the masked csv files.
   - `postprocessed` contains the output of the persistent homology calculations from `perseus`.
 - `src` contains the main scripts for the computation.
   - `make_dataset.py` contains the data wrangling aspects of the project, with
   methods for converting matlab files to masked perseus input files.
   - `landscapes.py` creates and manipulates landscapes for machine learning algorithms.
   - `permutation_test.py` contains a labelled permutation test.
   - `svm.py` contains an sklearn Linear SVM.
   - `modality_labels.py` contains the true list of modality labels for the experiment.
 - `main.py` contains the main scripts used for running the pipeline.

## Workflow

The main workflow of the code is as follows:
1. Pre-process the data: convert matlab files to plain text files, apply the
   mask, and convert into perseus input files.
2. Run perseus on the input files.
3. Convert the perseus output into a numpy.ndarray to be fed into persim.
4. Using persim to compute the Persistence Landscape of each time slice.
5. Label and process (pad) the landscapes for the statistical tests.
6. Apply one of two statistical tests:
   1. The permutation test. Given two modality labels, compute the average PL
  of each label and then take the sup norm of their difference.
  Shuffle the labellings, compute new averages and the new sup norm difference.
  Compare this to the original differnce and determine if the shuffled labelling
  is significant.
   2. SVM. Given two modality labels, construct a linear SVM for each label
  and validate it using 10-fold cross validation.

This process is outlined in `main.py`.

**Note:** The raw data files are large, so the output of the perseus files have been uploaded to `data/postprocessed`, so the workflow can start directly from Step 3.
