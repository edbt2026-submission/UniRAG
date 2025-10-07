# Step 1: Prepare Files
1.1 Place the `.txt` files in the root directory of MyRAG.

# Step 2: Run

## 2.1 Split Documents
`step0-0-document_splitter.py`

## 2.2 Analyze Text Segments
`step0-1-0-segment_analysis.py`

Remember to uncomment `analyze_text_large`.

This step uses the official DeepSeek analysis, which supports JSON mode.

However, the server may be congested, in which case you can switch to Silicon's DeepSeek, but it does **not** support JSON.

Also, the `analyze_text_large` step does not generate logger output, so it may run for a long time without any visible response. This should be fixed later.

## 2.3 Check Analysis Results
`step0-1-1-extra_conversion.py`

Results are saved in `{filename}_extract_result.jsonl`.  
After the first run, you can check that the `entities` of each element are still strings and have not been parsed.

The purpose of this step is to verify that `0-1-0` has correctly generated output. Proper JSON will be converted into a dictionary.

Pay attention to the output of this step.

If errors are detected, the invalid results in the extraction will be removed.

**Important:** at this point, you need to re-run `step0-1-0-segment_analysis.py` to automatically fill in the removed errors.

In other words, `0-1-0-segment_analysis.py` and `0-1-1-extra_conversion.py` are run in a loop repeatedly until there are no errors.

Finally, check `{filename}_extract_result.jsonl`. The number of entries may decrease, but all `entities` should now be dictionaries.

## 2.4 Convert to Format Ready for Merging
`step0-2-node_file_conversion.py`

Results are saved in `{filename}_EX.jsonl`.

## 2.5 Perform Local Node Merging
`step2_0_go_merge.py`

This step may take a long time. Please be patient.

Results are saved in `{filename}_merge.jsonl`.

## 2.6 Merge Validation
Filter out invalid merges.

`step2-1-validate_merge.py`

Generates `{filename}_final_merge.jsonl`.

If invalid merges are detected, the step will theoretically handle them automatically and re-merge. No manual intervention is required.

This step may also take some time.

## 2.7 Create Merged Index, Prepare for Database Import
`step2-2-node_merge.py`

Generates `{filename}_EX_pro.jsonl`.

# Step 3: Neo4j Graph Construction

## 3.1 Start Your Neo4j
Remember to modify the configuration in `Neo4j_tools.py`.

## 3.2 Import Nodes into Neo4j
`step3-0-import_nodes_to_Neo4j.py`

After running, you can log in via the web interface to check the nodes.

The `DeleteALL.py` file in the same folder can be run to delete the entire database in one click.

## 3.3 Local Merge (Write to Local File)
`step3-1-local_merge.py`

Generates `{filename}_group_fusion.jsonl` — the list of entity groups to be merged.  
Generates `{filename}_group_stay.jsonl` — the list of entity groups not to be merged (because they contain only one element and don’t require merging).

## 3.4 Merge into Neo4j
`step3-2-write_local_merge_to_Neo4j.py`

Adds properties to all Group nodes.

## 3.5 Global Merge
`step3-3-global_merge.py`

Generates `_group_before_global.jsonl`.

When debugging, remember to comment out the `reset_jsonl_file` method.

## 3.6 Write Global Merge into Neo4j
`step3-4-write_global_merge_to_Neo4j.py`

Visualize the graph.

## 4.1 Vectorization and Database Storage
`step-4-1-Embedding.py`

## 4.2 Save Vectors into Local Vector Database
`step-4-2-Faiss.py`

Build the index using Faiss.

Install Faiss first:
```bash

pip install faiss-cpu

```
