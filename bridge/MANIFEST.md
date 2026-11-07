# Manifest

## Entry points (in order)

| Step | Command | Output |
|---|---|---|
| Layer selection, WikiText validation | `python pb_run.py sweep70` | `out_pb/sweep70.*` |
| Layer selection, corpus-free | `python pb_run.py sweep70free` | `out_pb/sweep70free.*` |
| Main fits (hand-over and comm objectives) | `python pb_run.py comm70` then `python pb_run.py comm70self` | `out_pb/comm70_*.pt`, `comm70*.json/.npz` |
| Donor controls (v3) | `python pb_mismatch.py comm70 pythia-70m pythia-160m 2 1 5` | `out_pb/comm70_swap.npz/.json` |
| Decomposition + bootstrap | `python pb_decompose.py comm70` | `out_pb/comm70_decomp.json` |
| Table | `python pb_tables.py comm70 ../paper/tab_comm70.tex` | `paper/tab_comm70.tex` |
| Per-position / truncation | `python pb_extra.py comm70 pythia-70m pythia-160m 2 1` | `out_pb/comm70_extra.*` |
| Receiver-state reference vs depth | `python pb_ceiling.py > out_pb/ceiling.txt` | `out_pb/ceiling.txt` |
| Implementation checks | `python pb_checks.py` | `out_pb/checks.json` |
| Hand-over diagnosis | `python pb_run.py diag` | `out_pb/diag.*` |
| Larger Pythia pairs | `python pb_run.py sweep410_160 sweep160_410 auto:comm410_160:sweep410_160:410m:160m:0,1,2 auto:comm160_410:sweep160_410:160m:410m:0,1`; `python pb_mismatch.py comm410_160 pythia-410m pythia-160m 6 1 5`; `python pb_mismatch.py comm160_410 pythia-160m pythia-410m 4 1 5`; `python pb_decompose.py comm410_160 comm160_410` | `out_pb/comm410_160*`, `comm160_410*` |
| Text scaling | `python pb_run.py textscale`; `python pb_mismatch.py textscale pythia-70m pythia-160m 2 1 5`; `python pb_decompose.py textscale` | `out_pb/textscale*` |
| Cross-family pairs | `python pbx.py gpt2_pythia160 openai-community/gpt2 EleutherAI/pythia-160m 0,1`; `python pbx.py gpt2_qwen05 openai-community/gpt2 Qwen/Qwen2.5-0.5B 0,1`; `python pbx.py opt125_gpt2 facebook/opt-125m openai-community/gpt2 0,1` (with `out_pb/x_lean.txt` and `x_skip.txt` as kept); `python pb_decompose.py x_gpt2_pythia160 x_gpt2_qwen05 x_opt125_gpt2` | `out_pb/x_*` |
| Irrelevant text vs latent | `python pb_irrelevant.py` | `out_pb/irrelevant_contrast.json` |
| Sender comparison (paired seeds and donors) | `python pb_sender.py` | `out_pb/sender_compare.json` |
| Out-of-domain | `python pb_domain.py code comm70 pythia-70m pythia-160m 2 1`; `python pb_domain.py reviews comm70 pythia-70m pythia-160m 2 1`; `python pb_decompose.py dom_code_comm70 dom_reviews_comm70` | `out_pb/dom_*` |
| Probe-text variants | `python pb_run.py variants`; `python pb_mismatch.py variants pythia-70m pythia-160m 2 1 5`; `python pb_decompose.py variants` | `out_pb/variants*` |
| Capacity | `python pb_run.py capacity`; `python pb_mismatch.py capacity pythia-70m pythia-160m 2 1 5`; `python pb_decompose.py capacity` | `out_pb/capacity*` |
| Generalisation table | `python pb_gen_table.py ../paper/tab_general.tex` | `paper/tab_general.tex` |
| Domain window identities | `python pb_domain_windows.py` | `out_pb/dom_*_windows.json` |
| Population audits | `python pb_audits.py` | `out_pb/general_audit.json`, `out_lm/probe_overlap_audit.json` |
| Text vs bridge per transmitted position (Bonferroni) | `python pb_bandwidth.py` | `out_pb/bandwidth_test.json` |
| Figure 1 | `python ../paper/figs/make_fig_comm.py out_pb` (alignment helper included in `paper/figs/`) | `paper/figs/fig_comm.*` |
| Paper | `cd ../paper && tectonic -X compile main.tex` | `paper/main.pdf` |

The run scripts actually used are kept as `out_pb/queue*.sh`. The small-testbed harness (`harness.py`,
`run_toolfree.sh`) additionally needs a sandbox directory (`BRIDGE_SCRATCH`), per-run task files
(`sandbox_task*.md`) and machine-specific paths; that set-up is only partly documented. Run Python with `-I`. Hardware: one Apple M5 Pro (Metal backend, float32). Environments: `requirements-lock.txt`
(analysis, Python 3.12) and `../paper/figs/requirements-fig-lock.txt` (figure, Python 3.9). Every model and dataset
load is pinned to the revisions below through explicit revision dictionaries (`pb.REV`, `pythia.REV`).

## Revisions (local Hugging Face cache refs at manifest time)

Models and datasets were loaded from this cache; it was not updated during the study, but these refs were
recorded afterwards rather than at each run.

| Repository | Revision |
|---|---|
| datasets/HuggingFaceH4/MATH-500 | `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` |
| datasets/Salesforce/wikitext | `b08601e04326c79dfdd32d625aee71d232d685c3` |
| datasets/google/IFEval | `966cd89545d6b6acfd7638bc708b98261ca58e84` |
| datasets/openai/gsm8k | `740312add88f781978c0658806c59bc2815b9866` |
| datasets/stanfordnlp/imdb | `e6281661ce1c48d982bc483cf8a173c1bbeb5d31` |
| models/EleutherAI/pythia-160m | `50f5173d932e8e61f858120bcb800b97af589f46` |
| models/EleutherAI/pythia-410m | `9879c9b5f8bea9051dcb0e68dff21493d67e9d4f` |
| models/EleutherAI/pythia-70m | `a39f36b100fe8a5377810d56c3f4789b9c53ac42` |
| models/HuggingFaceTB/SmolLM2-135M | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` |
| models/Qwen/Qwen2.5-0.5B | `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| models/facebook/opt-125m | `27dcfa74d334bc871f3234de431e71c6eeba5dd6` |
| models/openai-community/gpt2 | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |

Code domain input: Python 3.12.14 standard library, 165 top-level .py files, SHA-256 f17cc33b8c17e0c9.

## File hashes (full SHA-256)

| File | SHA-256 |
|---|---|
| `audit_noinput.py` | `42fddfd7d2490d74a8805f062459402cedd5ad3077529bf18457cb829f338f70` |
| `common.py` | `48859aa9722d06faea3fa9fb23e4f677be209441428d99b8839be33e1f251e7b` |
| `harness.py` | `59bd36d9deeb2d322dc09e83112594d383df257bc2408aa66c1c26a770227942` |
| `lm_baselines.py` | `a35a879da17888f1e2e62dca58908c06d899d4596c3797aa05fbc8c4f9986653` |
| `lm_common.py` | `540731c31aa0d837b46eada3284deadc7e03defbcf92d95e1fc41b4d06387c95` |
| `lm_score_llm.py` | `ac8f914b281596f0ec4502a0f1cfd1f5f56d0c16e8d60d7aa8552e5199d950e7` |
| `lm_seeds.py` | `99d2189422d7e17baf283101703a4cb30678006fc3ad9d4b5e680a5dc3e4f2bd` |
| `lm_selfsample.py` | `79fb1e6f17b61439b98a31c48cd99ad94c47d077c76718d17a7fe212148b54dc` |
| `lm_taskfit.py` | `4359c4d0d81c6d67123a2c9d3a11b0a9357dee564f7e728e6d03eb819a5954e2` |
| `make_manifest.py` | `c95cca0d7cde6e44d25ccf1b8aba273616866a5ff9925975d5ae809a2caf8a29` |
| `make_prompts.py` | `e9ebe08e2bf03bb63faea44f06de7b34e5fbe80b0097eae4919a5f81ec928a2c` |
| `noise_distill.py` | `07517c11e2fb04aa25d970678aaf2db21e7e16e38b63246fe33681a956c069c4` |
| `noise_probe.py` | `51e6fbd85c075de468275d8ab2f4029667b3522b9fde6b0ac932ee6f1aa65e87` |
| `pb.py` | `6e38cb7595e909ab466441095d28cb370514768c28f486d445822be4de76a38b` |
| `pb_analyze.py` | `f7a792bea9e61ed51a48e73c679f392bdcf3eb3eec42cbee319c3d0e933e145a` |
| `pb_audits.py` | `8e054906e4dc0a13c23ce765c1b1d0834c30fccc0920c817e0a3aed1d311e834` |
| `pb_bandwidth.py` | `36894c3169ff47ddb513cdf4fe6d8c1b9ea853feba73146d4f6628611308fb8b` |
| `pb_ceiling.py` | `39e1ac358082e72ce664cc699613651ac4877b9aa66782e2709f821125a1ecfb` |
| `pb_checks.py` | `86b8f385038cde0305a4a08bfa0db95e7ffc54d1b4eccd23facd299203c8232f` |
| `pb_decompose.py` | `261be7033daaa1d41ce82f2db6bb77686a4ac07ac29422178fd294341d001c66` |
| `pb_domain.py` | `97dfa0f5d584b0ee1ae5ceb76d0219e50bd99aaf29fa3ff7b5b2d4f16dfcf1d8` |
| `pb_domain_windows.py` | `c69aca364159114b3d12ba3ae2e1e40840b016469d973baa78cd663d8daf6c78` |
| `pb_extra.py` | `197ee12c63cd193d1115961a4bfba48196941c5f5cea216ff82b917355a6af5d` |
| `pb_gen_table.py` | `6c1b7e698014b012dc50bcef770abc8374e2db9775c3d6147fe07772a2476f56` |
| `pb_irrelevant.py` | `0fb270f951202e29f472e6f5737d663eceffea2911214fe53730fa293dcb45be` |
| `pb_mismatch.py` | `42e56b2a9ac35206b9fc7426abde09537a797ef68bd5b10980990ded43dbc3d4` |
| `pb_run.py` | `3a51731a0ab059125281f1f82a09ed501543aed71858db4f5e42df6c4e7274e2` |
| `pb_sender.py` | `abde8bfac9a5457ee909d5f278d7a57683c527bf00605e2e9c3bb29b8fdde820` |
| `pb_tables.py` | `c56f6ee56a621fd007fbb5c81304daa002ac94142135c0629ed0850f31cd1cf1` |
| `pbx.py` | `d9a9fa63eea66dfc719ea0de210a491a08178ac2136c41362c59753a19423cd4` |
| `pythia.py` | `56bb830851b3910d06fe11f1a45f38c3324b06434e85dcb5f60d2d1a4e6bdb9f` |
| `pythia_probe.py` | `9dfc6b667da3a3c90d8331cc21d772d0bdfcc690ced0540c7cb401cd750c99de` |
| `pythia_seeds.py` | `d547534608c21b5246eb4c6b22bb62ba7f79a65fd56067ca1b2599eb6f8155c9` |
| `run_baselines.py` | `58994ad12d0ce52ec48cef71e63523f94ce3b35cc1286a9ac3ccb01f7434160a` |
| `score_llm.py` | `ccceebe8bd3d1272ea34ecdcf7109bfe2d344b17e3442c26f3a5be660056338e` |
| `summarize.py` | `856aaa1eff7ae88843ef0340aceae3c4100e03addc5ed55240b2a8ad4c3c31c7` |
| `run_toolfree.sh` | `facfabaa9f7e86fc585d6c99217f3c750676674222b668c85015431d203ba9e1` |
| `sandbox_task.md` | `d20dec5e72104cceba505d5a30df5139a815e1152153acb003857d86181b28b0` |
| `sandbox_task_lm.md` | `cd4b03d7d31419d18be0374655356e42a586697b1fde0879dd089bb9ae221748` |
| `requirements-lock.txt` | `03354d2875d2ee87c4ee7b9ad5a5154a0cb0e587592241975cf0deafd498b6b3` |
| `out_pb/bandwidth_test.json` | `938ec4750a535f0438e9512758efb1754c435fa77d42425584b898021bcb4ab8` |
| `out_pb/batchroll_comm70_extra.json` | `def172b1ef3470f07658f8e58d50a37d5e10a4be6eb69c8c9b09f783a1829c23` |
| `out_pb/batchroll_comm70_extra.npz` | `e0b355e74066a1206ff353d359c55c587d8c6e0f59efb04dbc02d1ea9f81775d` |
| `out_pb/batchroll_comm70_extra.stdout` | `5f7ccf59001f0a80bd2020c8f90587b049588e2695d93579e5218a709cc57482` |
| `out_pb/capacity.json` | `296afa55b6cd69411a578d8e64fa66f2909976a731d2049418df510e6827bf17` |
| `out_pb/capacity.log` | `210170c87509deb3b9e4453d8c50b415097af039dccb4f78ee57f5170552763a` |
| `out_pb/capacity.npz` | `531cc8a60e582b1a0fe4884c483f983d47b2fd1dd70676929c72c6296e0d4909` |
| `out_pb/capacity.stdout` | `9a53a6801576c2ea9f55acfc3d336cf768e1e8cbdbc2be955b4bdefa83fe5611` |
| `out_pb/capacity_comm_Claude-mlp2048_s0.pt` | `ccfd8afdb84082e5b12bea368d3266493db1d427b11daa1b55f50a3d74831cff` |
| `out_pb/capacity_comm_Claude-rank16_s0.pt` | `e3c0deea63278b4669d6d28c8a759be3a5794a76bfcdcc0642dd19c27137093b` |
| `out_pb/capacity_comm_Claude-rank64_s0.pt` | `0fdb5b5ed7b69799f423458ccca76947a1e037db5e0ab160cf9f43fc994f5b3e` |
| `out_pb/capacity_comm_real-mlp2048_s0.pt` | `37fa071d29e84235371de06aa1d0f73b8f09312c70bba53aa20e1c5fef92b126` |
| `out_pb/capacity_comm_real-rank16_s0.pt` | `fcc38fc9a3eb98b3b67da6ea1eaf86324efac8116041f31b16821639821fcf3f` |
| `out_pb/capacity_comm_real-rank64_s0.pt` | `66c5718bea97f8ca22e53900bd2addcd4f106f125f0f4c9191b1ba9d0130679d` |
| `out_pb/capacity_decomp.json` | `5aebaec36ed17c014d5aad95167ce36b66b6b754de0c9bec85380cc95bc8e3a1` |
| `out_pb/capacity_swap.json` | `4f40a1aef2f7b4371efd224fb6ace18d368b9ca4f4c2f1391eb784d67c6f0268` |
| `out_pb/capacity_swap.npz` | `3931a73688484e230b09508778b201a4554c2eff9b950bfc9f4b7e1c1f4bbcf4` |
| `out_pb/capacity_swap.stdout` | `8d467e1ab40142505b32bb8a5329fd549f605a4ce20e2ce9a617e0668ef1d46e` |
| `out_pb/ceiling.npz` | `15069016bbab83c8e202af292e41088e5188cc455253a549a8cca43481be2435` |
| `out_pb/ceiling.txt` | `f910c284603b12bddc411a80720ac8e23e77bc1c185cd80c659a78ec9156bb2f` |
| `out_pb/checks.json` | `653aab9bbb41f7ea078b46bcb61b53b07fc5018be50b1ff5b27aed7702cd78e9` |
| `out_pb/comm160_410.json` | `e8378e8f197927ecac2d01ff7a5cc4f94bb819925894a451e975e99446d66d9c` |
| `out_pb/comm160_410.log` | `978ed479315c9979cba4806005f8b589691bc9c2844ed14301e2972b8df1b1b9` |
| `out_pb/comm160_410.npz` | `6b3ec3b6e797945f2e3fa37c7b40fb8a8154ea98a1142e0f86355deb0094d1a8` |
| `out_pb/comm160_410.stdout` | `ef86dd4469ff28c57650418e2e96e19a4859f5fba11024a5c14bdaf9d00753f0` |
| `out_pb/comm160_410_comm_Claude_s0.pt` | `2ee1ec059e6240c7912435a9f0bd49bdaf4d8296cb30f68e48734338c3e9aee1` |
| `out_pb/comm160_410_comm_Claude_s1.pt` | `2b6a6215ad01b362926d4062ae704341dacba766bfa5bbd6115f7691e50f90a8` |
| `out_pb/comm160_410_comm_real_s0.pt` | `7c24568e3132cd4d29da1713890e204bfae880516bef534689407e54f15d2830` |
| `out_pb/comm160_410_comm_real_s1.pt` | `ca9bda3aafbd548d4e375af5a65ad11d575d2bee16981dad1a69606b6df2e82a` |
| `out_pb/comm160_410_comm_self-sampled_s0.pt` | `05c07a354eea7a51edae92f47bd5c65cb78781ae06fded8da1e833fba43bc288` |
| `out_pb/comm160_410_comm_self-sampled_s1.pt` | `6e1bef2df698cbee3e69c79d1ec035b747739a325868dba2371f532c63b9247d` |
| `out_pb/comm160_410_decomp.json` | `1e8ef41f03b78222e41330bb7429ed37fef6171179f968e5ae16ade12b691c25` |
| `out_pb/comm160_410_swap.json` | `4f4bdaa5e6f82ee0fca7035fc90a9d7835b2706825a436de8d0b33ce03d83b0f` |
| `out_pb/comm160_410_swap.npz` | `8c2bda1e0cb4d32c750647c3e6309741cd93c7bba5c94da7df72af93f19c08c3` |
| `out_pb/comm160_410_swap.stdout` | `66d0c18996743405e53fcb41717be93157bad9e9576a3bd068337a0afab15d5d` |
| `out_pb/comm410_160.json` | `5901c5e7a99d8cc61dec304bd6098028e873a500bd42dfd51da4ff8a034411ae` |
| `out_pb/comm410_160.log` | `7cfc9f77c761f49ce0826c1027f78c26c2a25c8f83d7048a7149cf0007c78a8e` |
| `out_pb/comm410_160.npz` | `ba02f3fba41815d7d63bedc941d08853f263d14fafdb4b9b955e95d82939014b` |
| `out_pb/comm410_160.stdout` | `cd9820654be43619baac4e22c3417d9010d4d27b717e27d596db1dd93c01ff90` |
| `out_pb/comm410_160_comm_Claude_s0.pt` | `4b37a96f73c3b198ac8d1466078859084fb22003ea1b7bf74f5dd21f0da1012c` |
| `out_pb/comm410_160_comm_Claude_s1.pt` | `43ed2d24cc789a48c0cc897ebf230044667a44515431bf178642ed142a2a2c4d` |
| `out_pb/comm410_160_comm_Claude_s2.pt` | `d27e5c79e5086a8bfced14fd1195195f12caccc80d0eaf602733f4f7d7e93b31` |
| `out_pb/comm410_160_comm_real_s0.pt` | `07ced0bcf2507cea04bcb3ed3df645d3860ff8531adafd4e571c34bccd54ab1c` |
| `out_pb/comm410_160_comm_real_s1.pt` | `e86b06261f73fe0ea6af511def62fddcfebf27644d1227923fe61190e735d7b8` |
| `out_pb/comm410_160_comm_real_s2.pt` | `49cc319036456ed7b2df633269e20997f4053af7eb9dd6d43a44169a89fa984a` |
| `out_pb/comm410_160_comm_self-sampled_s0.pt` | `7cbbb406259aa6163f7d419fe18580ff675794edd4c5976bd1d837b679d89573` |
| `out_pb/comm410_160_comm_self-sampled_s1.pt` | `848bea6c427d949749918cbd0edf041667e921109596171f58362097b67bbb57` |
| `out_pb/comm410_160_comm_self-sampled_s2.pt` | `825257fc6f25dcb5b4a539867577087cf1da1efcee5572cb0e8669987c0c6958` |
| `out_pb/comm410_160_decomp.json` | `3e3585722e646d47f3723779a10939cbe40d9c20a8c3a0558601d3c77913e368` |
| `out_pb/comm410_160_swap.json` | `aabbf9386edf0f0d50d08b298d98df2918192d30aabdb38b11bb1b3abc84efd1` |
| `out_pb/comm410_160_swap.npz` | `1c904454aac11b44cb472278c4ec2cac52f99fd0dd52b2522e1752d93ad56b16` |
| `out_pb/comm410_160_swap.stdout` | `cb66952191b3a749bf40d198a9c5141846cce76a4dea0e4c207b1fc0141f7f9f` |
| `out_pb/comm70.json` | `60afcac1910c40da3f2c830ee230474803e7fbe1694572207714daa13c0c0d50` |
| `out_pb/comm70.log` | `1317ef9ccc7a5514a59edde1f17721f79b143ddefc6b47fd594427d459147346` |
| `out_pb/comm70.npz` | `69a711db43834453084f4413f8273c79982c0c996efd5a4345e9f736ea5ba825` |
| `out_pb/comm70.stdout` | `6f973c8c1dd4ccd018a9a1944c3ed2e6b664443718af3ad52b0fb183b5f8b11d` |
| `out_pb/comm70_comm_Claude_s0.pt` | `8c967b8a6b8ef75bba01b95c86ab992c7527349ac21bde756a571603e21d0762` |
| `out_pb/comm70_comm_Claude_s1.pt` | `97584a45af36bc59b285046a6c19beed1ef146d4515d5e495296a8ab88cf0279` |
| `out_pb/comm70_comm_Claude_s2.pt` | `216ac72b4555d9a8229b4bde4bfeb93e8accb9734bc2b012287801144ea4e928` |
| `out_pb/comm70_comm_real_s0.pt` | `df2cac871b0aa7aea3f4c34e90a8b0a5197a60cb1408075d84ef046a9c15f127` |
| `out_pb/comm70_comm_real_s1.pt` | `f82120ee47d462c2ad81d39bd958cfbf790456ffdba38b929c89abfcdbcdb3e4` |
| `out_pb/comm70_comm_real_s2.pt` | `066ba12dbb2db36ae2062efeaaebc4a40c96fb8d25df371450804c2912cd802b` |
| `out_pb/comm70_comm_self-sampled_s0.pt` | `a21eb4c385e65de1174a9f58dcfcb84fffbf3ad5d9b06274fa0c6a18ae56bce8` |
| `out_pb/comm70_comm_self-sampled_s1.pt` | `08e5ad78f3e5771c2266cd3873a14eb78f9e37bf6fa5d049c4a18ac1418abc8b` |
| `out_pb/comm70_comm_self-sampled_s2.pt` | `0f1d9b6555f0b616cec424fc71b9e5026b63a3e472923b0ea1c67d40a2383c86` |
| `out_pb/comm70_decomp.json` | `62b51e2c00372fd5750c02261b2f1c9da4774ac6cf51f199ee38ed714d4cb1b8` |
| `out_pb/comm70_extra.json` | `b221ee8b9e14f192c0d08b02d7c79e46595fd27bc82029454876fca9755c0657` |
| `out_pb/comm70_extra.npz` | `6fc55f3ab3b9603e51284f5a4f16d98904d542f0a73c042bedb35c48037ac1d4` |
| `out_pb/comm70_extra.stdout` | `2e5225a8b37206958dd83e2b364950cafe2c222f9b626a1743c194a42e2661de` |
| `out_pb/comm70_full_Claude_s0.pt` | `2d45df31783d6d245b5bdddab305053b742b9d9a75507dd5e20a530f26e82e55` |
| `out_pb/comm70_full_Claude_s1.pt` | `a62294d4158835e8faad0d13bccbf31f5410c3ae24ea8613026618f5a479d0d7` |
| `out_pb/comm70_full_Claude_s2.pt` | `84ee5e1ef01395d455a2a363741ea752cf2e314da7ca83c99e4ad7f5a2c8c00c` |
| `out_pb/comm70_full_real_s0.pt` | `3559dbde5b35b5ae64728e5699c13e01466bf4ae3bbbf605e074f39dbf5ff574` |
| `out_pb/comm70_full_real_s1.pt` | `535883a6d2bc199743910bcb266f48656faf1d9e7f76ba1f3df862f4fee097b0` |
| `out_pb/comm70_full_real_s2.pt` | `47f9e09f7eb3322a539151102e64c1e4548b4cb8a999618909674d489062f487` |
| `out_pb/comm70_full_self-sampled_s0.pt` | `f7576f84646b410b827768e0b364a90e76ae96733efc741cfccad52199077df9` |
| `out_pb/comm70_full_self-sampled_s1.pt` | `5151bbd05cea77bc6ba46cbb963388347a4e048cad4b5300cec14ca9db0a3ac0` |
| `out_pb/comm70_full_self-sampled_s2.pt` | `f27b200e7358e58f25d6275dfdf4138baa0036d28fc3905319ae90371f209afd` |
| `out_pb/comm70_mismatch.json` | `35deb25f599233bdc2cf2b923a258f39e8c18f04707262f2b14d6e3188e5675c` |
| `out_pb/comm70_mismatch.npz` | `8fb26d1431ebc147b9d5ec9424afcd2710ed21291acc93bc72c922d7f5de16f6` |
| `out_pb/comm70_mismatch.stdout` | `0c7e5399557b62dde793291f87c97e6b52d55064c3e45d36a26b03989b145aff` |
| `out_pb/comm70_summary.json` | `55039deae8cea44e8d739d57cd9916882f9e0344b90bdde418cef07dd4bb2574` |
| `out_pb/comm70_swap.json` | `657e2b9e2391003f454ed508fea66f5875b2e1c48a00ecc9fe3230c75278e8bb` |
| `out_pb/comm70_swap.npz` | `77f868ec1adbecced61696e2c06c86f8d6a06d289ab5f24d47a62177012577d6` |
| `out_pb/comm70_swap.stdout` | `ff3daf222f2412b9cf1ad1c686a88fc0e2230a4560913926618415b901a7a4c1` |
| `out_pb/comm70self.json` | `a19a24be4a7db7d3aa73003acc5a50b22b3c763286f2dffbf1a7cb7f3bc094df` |
| `out_pb/comm70self.log` | `f6d32310aaea31eff70b9c50c2c318b3b2b111f3fb18539e6b43bc86d5532423` |
| `out_pb/comm70self.npz` | `3097a2b604b500ed612fd1adcb511d9852e132f5bf7f3787620efa1355b4a41e` |
| `out_pb/comm70self.stdout` | `4bf0f285816b25a20ca806f870673cc301fef06b020d1084439f33b11d3448e3` |
| `out_pb/diag.json` | `007a08ccad18887fe7d193196996165d83364fdd2058dc6a3212032da8655a26` |
| `out_pb/diag.log` | `d45ae7add5c2e8e2bf23364f7a4296a91a5ab9feef7da5140b6e3d3b2666dbfd` |
| `out_pb/diag.npz` | `ab429ed5e4c280695744b2ec852335f2c983ff3e876dd3eee7374e984bf882fb` |
| `out_pb/diag.stdout` | `5c60cc99f70c18f4e154fec6a246c87b5cdfc79fa35e4331a30fe4ba2d14fb21` |
| `out_pb/dom_code.stdout` | `a5cc10e84f059a4e771e8803db58047ca6ee2a215865bf6daf49c1a05db64e3d` |
| `out_pb/dom_code_comm70_audit.json` | `79b67187e150884634a3b4726a9457fe202856f449128dc4a674fc51dc400b54` |
| `out_pb/dom_code_comm70_decomp.json` | `3779b7d4876d1b1b7d32ee8aa07dc3ac25db0ee5dc09d036a9dea4a93aa9d4cc` |
| `out_pb/dom_code_comm70_swap.npz` | `94737915b5a0c9ef8e9ddb978dc38447563c231c7f56f76a08beee8307e9a9e4` |
| `out_pb/dom_code_windows.json` | `49ecc951b3d8b001e2ff8ccfabf9d4df5d3aa4924c8be0cdae0e55583180a85d` |
| `out_pb/dom_reviews.stdout` | `81f52aa4947591352cb5adc3ae428bd077f90a6286ee624e547c2fd7ef9abf58` |
| `out_pb/dom_reviews_comm70_audit.json` | `bff2e936b5ad5bf00066f826c63d841a9f7ddf4caea33429f2929cf26927568b` |
| `out_pb/dom_reviews_comm70_decomp.json` | `43cf9bd5e8ed2a903b17e66b8d209d4752840c4010db93a9e93c34d2a104c401` |
| `out_pb/dom_reviews_comm70_swap.npz` | `a92bb5b34b8e554df8fb3b4991df63ded8fa93b9a2202a61e0d30b9d494973a7` |
| `out_pb/dom_reviews_windows.json` | `b2a9f84f09f213108985945366f2e2537280a6599c39ec86d24f870114d189bd` |
| `out_pb/farshift_comm70_extra.json` | `b1ec4b2b26a5940b9e43b921187ad2b654c12f3ef0a356c38cd94787735b62c5` |
| `out_pb/farshift_comm70_extra.npz` | `2fb71e07d90c45dbf7c0b7bd8a7652d7cfa4840890d5323fb1e6823211c64235` |
| `out_pb/farshift_comm70_extra.stdout` | `70a2e314c114cd03960ca50c267f0f6620ffc34bd228808040c50bd43f1dd571` |
| `out_pb/general_audit.json` | `73b538383182fc0ce6d3fd8688161b7370fc23aeef8b453d5e8d833c8eb5e431` |
| `out_pb/irrelevant_contrast.json` | `a1619bdc069ed67944444b4d8696d9d5e792f0c3e2bd0cf3b1c8cd81334894e7` |
| `out_pb/overlap_audit.json` | `a881227389c492185975d5a1b72879248d1935f95e1a7aa94d50a108be9b8367` |
| `out_pb/queue.sh` | `e2af168d75ba08bff3419cd0f86afa99c5f52fd3f2180980749ef8cbc54312e5` |
| `out_pb/queue.status` | `b0a57796301dbb3daf9c9b7fdd04ece439effbf2f116bfcd1c9a3f7235fc6f97` |
| `out_pb/queue10.sh` | `4afdbc2f51e17042272cb33bb844855abdadd285217e4e5ca0536a637722622c` |
| `out_pb/queue11.sh` | `919602d8d6f9fa6a0a600a37a36aeb48284d75f1b726cb4d1c30696a34c65885` |
| `out_pb/queue2.sh` | `6530a7bbd5e8c0f1fa0ef90d5d3fec0c98262ed95fb4f55164e4560546aba370` |
| `out_pb/queue3.sh` | `29fb970e867948859b5bc6c92b8707368306a0d6c8098379d971bac5eddcbc47` |
| `out_pb/queue4.sh` | `5a0c8d5ce0a269b7ce531ea87eab1c50f780e037d3cc6cfc7df76b5d9b7c768b` |
| `out_pb/queue5.sh` | `08960f686333bd53c57ea298089fc08f0b721b997f40853488735888324b19fc` |
| `out_pb/queue6.sh` | `c20c17b514c3c3147f324a951f5a24ffaa9996520edbb8f6e5ef23121fa44927` |
| `out_pb/queue7.sh` | `fa2b35d11fda6c33bea5692666e31351ce6506136bec8e01950f3928ea1b3bb3` |
| `out_pb/queue8.sh` | `3f6d9ab4106d4e805855e267e619adc3b043a20ecd93711ba294715709539643` |
| `out_pb/queue9.sh` | `fa593dc3c2893113eb17e1a265d1d29a9948aa7d3c0a7ad48c49fb31b083bb0b` |
| `out_pb/sanity.json` | `9f3398cadd0c0e7f6f5ef5b91aa4993c663557c8b6a204002e09865c9a5dacac` |
| `out_pb/sanity.log` | `813b8ce028920a900f4e0931d3363057d3217d3369780895d5f3685f51bfb3f3` |
| `out_pb/sanity.npz` | `ea8c68f94682e1adc013faaedacebd022e3cac5f6eef3dc52b80b4b99e8784ef` |
| `out_pb/sanity_summary.json` | `9d0bd54d5d43698bd5ad6090e244857d631595be1e6d43263637a91148a2e1b5` |
| `out_pb/sender_compare.json` | `d3f9bfb70d7e8af388c29660f186fb8c6480f78e1a999735ebc0b510158f936c` |
| `out_pb/sweep160_410.json` | `0edad3fd7b7ab49e6815842a53306053bfc999cc27e090c18b5f24f87bc769cc` |
| `out_pb/sweep160_410.log` | `914e0a39511a8d933df6230cb6716dd4eb38dea035c0ce23beb3c8d6cb2096fe` |
| `out_pb/sweep160_410.npz` | `a7a3c7d8946d0c279de2a37a2dd84a44b3fe169586bd14628d3e7d37ef8cb845` |
| `out_pb/sweep160_410.stdout` | `4632ac2aaadabe9eecffe6a01eadc1fd8100d468b766a9e6c66513813967e031` |
| `out_pb/sweep410_160.json` | `ea71b5a7cadfa62c98c9656ba388068f3528dff0944ccc7754110b825ed74487` |
| `out_pb/sweep410_160.log` | `401ba1a76d7f5a0a0718d2dcbad611e88d8828cb03cf855026d6866b190fd527` |
| `out_pb/sweep410_160.npz` | `6b7570121373976178b766591fd73651c71641134dc7e3fbcd32671970c1734e` |
| `out_pb/sweep410_160.stdout` | `88b7f8a4c11efa8a5073d9855d2d158ae41da72abaf03e053310b78f30f8d766` |
| `out_pb/sweep70.json` | `d88b5bf9395b47beef63efda449f5bfb8b9cbffd172008f13a8b9bf4fb7ea0b5` |
| `out_pb/sweep70.log` | `26f495b2bf1496ce7aeaa7ea448dfccba1d378d27e308459488aa2ed46c3050a` |
| `out_pb/sweep70.npz` | `d5e4e484f9c7540a3f24a3042d00812f33c824b31759bafd21136998f1b7cb8e` |
| `out_pb/sweep70.stdout` | `1930933420f9d0711e06f9ebec030b18f8dd86dbbc997f7a3363b1ac9ca388b4` |
| `out_pb/sweep70_summary.json` | `4c179726874b3903c6d5bf972cf2974350c39ed9014681a44627eba7ce26f671` |
| `out_pb/sweep70free.json` | `e8fe49741417ef38d1256b174988a3f5191fa5c2568295a16f220d78a84977de` |
| `out_pb/sweep70free.log` | `6346ca155ad182a118d6a999db5df3288c0aca04b5238a756713eec26f856d78` |
| `out_pb/sweep70free.npz` | `45500f1c37252c86b89d40a5e00a5fb1b97e032f0163cf29054c4b4632b2bf2a` |
| `out_pb/sweep70free.stdout` | `c7113c47793e180d91582788017dd74b10ef72170201128138857295ad45e395` |
| `out_pb/textscale.json` | `5f59281cbecb83a8de71ed1710c150e46b3d3b7f76c6a8223e884157d25e992b` |
| `out_pb/textscale.log` | `becf791a84fc5987db64598314d3a86d208e6343033e81fc962080338ed88891` |
| `out_pb/textscale.npz` | `8080e0ec81ff8c9a72b7fedcc16bca15912db0839694802b482704a66a33c365` |
| `out_pb/textscale.stdout` | `94de09f66c045d80d01fbf4de11056fb0882e9acd3399dd4c979640b650dc96f` |
| `out_pb/textscale_Claude_1030k_s0.pt` | `f8ea34bcc71a819606ae92dbb8edc6fe3f378728eca1ecc0d4b2cf38ecaed380` |
| `out_pb/textscale_Claude_1030k_s1.pt` | `77117a39a597648ed9397c55aef2e2dced625048f79a1f11547ccbfe963ea51a` |
| `out_pb/textscale_Claude_383k_s0.pt` | `a8cc4a6105754d57b91b33956738ed88ca6874136bfbc44573ba5877a1402abf` |
| `out_pb/textscale_Claude_383k_s1.pt` | `96f9b253987b991398f34c3fc4f68f5bf70f2abe1cd84b0014b26619b3b9cea2` |
| `out_pb/textscale_aborted2.log` | `aade84a97a0a34361ec65a81cabec58c2f1a2ff58876bf51160cc586f4217610` |
| `out_pb/textscale_aborted_batchroll.json` | `7751bb88c261ea6f11b3d3fa08487b2460c46fc33a69127afac41bf1dbf5a47a` |
| `out_pb/textscale_aborted_batchroll.log` | `af82cedf0c44dcbb6d0cd47424041b647871489ede918a6f5f56b6e28c5a1332` |
| `out_pb/textscale_aborted_batchroll.npz` | `83566c01a5daab00e6f60570bc6de602b35893eb2a3f881340db75028d37205f` |
| `out_pb/textscale_decomp.json` | `bf009b04ff32c0e70d610d22cb7a36e53421e70fe598c71a2a6c6227cbd7b8cf` |
| `out_pb/textscale_real_1030k_s0.pt` | `79c5a455e4b3c7e7c325ede6e340457b183048088cca0d1c3b0b402203646a9d` |
| `out_pb/textscale_real_1030k_s1.pt` | `7f3e03ff7e355b360aeeb084d12bbfd9275ba26050af4305c04721b857cb1cea` |
| `out_pb/textscale_real_383k_s0.pt` | `72f15dfef1df51c535c5089e46844fdd12b6d171445ce284c9a25de792129aed` |
| `out_pb/textscale_real_383k_s1.pt` | `23caf529363b5b0df5a9fb1b4d481c318717d7ff1a795fe357e4bf1a95032f85` |
| `out_pb/textscale_swap.json` | `7022c1881f8cccbec5a6df535ba3aad5a8c8d31740df6e9cda5418f36b0b5310` |
| `out_pb/textscale_swap.npz` | `1d97c6f69203acac1e5c556bfe62e3e559eaf0cf67dd3f0b1e88457fa2942774` |
| `out_pb/textscale_swap.stdout` | `07b1252789ff6d1e40c0c3ca1416d35c7a408b19529f3ba4c00d98d419349a0c` |
| `out_pb/unpaired_sender_compare.json` | `01cf5d582509c9236ee1b84cfa8ea47bc3d17ae01c1165605f953ce9503b8b35` |
| `out_pb/v2_comm70_decomp.json` | `b43af40b3eed97195b1d44d1eac48de0dc4e42b398dd6bcbd58c2069dabd3226` |
| `out_pb/v2_comm70_swap.json` | `9a025e0dbc7e47c449f5e552aa36f743c64c594c51575aef6de5311fed77b807` |
| `out_pb/v2_comm70_swap.npz` | `b80683c49edfeaca0bec727b0981ce7a00100c38c02b0ac336401b662ab52cb4` |
| `out_pb/variants.json` | `83a43d884bedeaf892eeb4b7deefe18dc2fa169507229710ad8457f791a9c8ad` |
| `out_pb/variants.log` | `596198dcf55027f38a0047bf20e6b55a42ce7bdc6b2c3aab5e3c76506555e146` |
| `out_pb/variants.npz` | `515fc6ef8ed09ee0b80696be286c56151b8479a173172238a85624f3d31a12f6` |
| `out_pb/variants.stdout` | `0d5b3210aaff4e0e09f355b0346ba04b83ee6d084946520bc77157bbef8a6afb` |
| `out_pb/variants_aborted.log` | `93019b6198f437d2da9816ac216d46f32bbaa530c9f3c96ec83f44dfad8fad59` |
| `out_pb/variants_aborted.stdout` | `12e5f46960525bda30c13f8fcf0c2591061b0e71ee9c332b856c7319c4519c55` |
| `out_pb/variants_aborted2.log` | `b5481fb5b74b6d5f26a7461b953c054930981de3543dd86b4547b2c76e13d413` |
| `out_pb/variants_comm_claudefiction_s0.pt` | `8e335c2e69f8f3879e11348b4f82d719ed253fe97d97d2d2b969303125e329f1` |
| `out_pb/variants_comm_claudefiction_s1.pt` | `5394e17969448ff9035c49fe4170eeb8476bea5c64151086ec7aba5cb1ceb9b5` |
| `out_pb/variants_comm_claudewiki_s0.pt` | `8c706653ceda4fdee5c332dc1e89d2610108c9f874ef7eed7b364c20a0ed6402` |
| `out_pb/variants_comm_claudewiki_s1.pt` | `f98ef46efa856d134082dd8471c5a32b62ab098e58c342e9e947ba4131e28570` |
| `out_pb/variants_comm_gptwiki_s0.pt` | `80ebf2ff77bd43b03d44d236297567ea5ecb313642bc85f67298c57d52e98c62` |
| `out_pb/variants_comm_gptwiki_s1.pt` | `540dfba36388629a46940b1bf6aae03efcd043bc3258e87f388ca8d9dffdb95c` |
| `out_pb/variants_decomp.json` | `626a890b907f457964e6345b9788cd81c667663160007f3684dffd1bbb9cd18a` |
| `out_pb/variants_swap.json` | `bfdbe4bab7517f16cb1f68dbec11af42a16574d4fc2823f504561aa242d93f93` |
| `out_pb/variants_swap.npz` | `cdd3a229251ab53178c9836347e0375c815dc36894a659dc97efcfc545dc913c` |
| `out_pb/variants_swap.stdout` | `530a2c06f9eec906b8b70761de62f668380f36dbe5295c01c10ab859b586fec2` |
| `out_pb/x_gpt2_pythia160.log` | `37dd080bb107556026e3340459b1b1bc1b99972e81196b57e383d2c162d5fe41` |
| `out_pb/x_gpt2_pythia160.stdout` | `2d5309cf20b43011cbbd7d0aafae6fcc34cc05178005d44485bdf80abc06de92` |
| `out_pb/x_gpt2_pythia160_comm_Claude_s0.pt` | `697577eb23e44e70f0fa79a22fa85d3fd45748b431485fa0620b1638c9c0844a` |
| `out_pb/x_gpt2_pythia160_comm_Claude_s1.pt` | `771a190de1e5ff202204933c031766479fe286aefa669c52ea669802ae7f26a7` |
| `out_pb/x_gpt2_pythia160_comm_real_s0.pt` | `3973154a774803192114ee2cd0c99c1685626c04d5fb6dc23357a015cfeeec6e` |
| `out_pb/x_gpt2_pythia160_comm_real_s1.pt` | `8222eacf5706870865c22017c0684156e50c8b736b0e1dc7c4c4060d1235aaa9` |
| `out_pb/x_gpt2_pythia160_comm_self-sampled_s0.pt` | `ee05d5a4189f0f606fdeea361bedadc8e5ca3732f15e4e7e0f7426609abeb262` |
| `out_pb/x_gpt2_pythia160_comm_self-sampled_s1.pt` | `a8c7011e9f323df10e10d691fd8a0b5af1946ea29bdcb149ccd2ac60c9222706` |
| `out_pb/x_gpt2_pythia160_decomp.json` | `acf887eb66bcdb30da0f3d6545148623139994cd81274b0fdc4d0ed34d4984c0` |
| `out_pb/x_gpt2_pythia160_meta.json` | `4d1219dcf1354b9452f10128e1845dc239ba0a89572721638cd77fa84a2f810a` |
| `out_pb/x_gpt2_pythia160_swap.npz` | `0ff2e86b6a8ee7452b8d7140efaa4785400d2ebec9be19ee8a4a2d6530cc2831` |
| `out_pb/x_gpt2_qwen05.log` | `2f6d3b857b1a23c9cd15ba3dd6ccd28e70c599bd65446e771a416765aa15eb51` |
| `out_pb/x_gpt2_qwen05.stdout` | `a6f4b9f750c1e2efa9124da0058ea09f05cb4a3cbfce361e216399930f450202` |
| `out_pb/x_gpt2_qwen05_comm_Claude_s0.pt` | `84c69d13fe06328abf1f4b745836b1ecedf8c3f6274a3f95432f567c4cdb5298` |
| `out_pb/x_gpt2_qwen05_comm_Claude_s1.pt` | `a0ca9c3366caee7052ae3a03b903950f5b7d8bcdb20219c699a79103d2ee34ea` |
| `out_pb/x_gpt2_qwen05_comm_real_s0.pt` | `1506a456989ed107570db9d152cbdc40ee8fedeafef797b55b017b3646ba87ea` |
| `out_pb/x_gpt2_qwen05_comm_real_s1.pt` | `f4e8960503cef1390b095243573fc0324d2a07d6ab63768d89feb25dbe4e6196` |
| `out_pb/x_gpt2_qwen05_decomp.json` | `b48a0d5b08554ee67fc4f2b4fde9cc9540887cb1eccfef5bdfeb6484ed1a5626` |
| `out_pb/x_gpt2_qwen05_meta.json` | `1d16f0fefb9de6cd09af90005754c8d786658afe5a646bf2158688c16ec15eeb` |
| `out_pb/x_gpt2_qwen05_swap.npz` | `c6d30f3cea85de10ecd4b66ebae69067070bed65605ddbdadbe7b94c7dcd54dd` |
| `out_pb/x_lean.txt` | `e8e98c2d7643b9d25c40828eae8e72845df3062fa65e858890063d8b2138e918` |
| `out_pb/x_opt125_gpt2.log` | `f5c66e77f25d5fe4703ae8801ff5983a303b2ce600e77fc08b0d5041fb95bdee` |
| `out_pb/x_opt125_gpt2.stdout` | `8801ad82df9e397d2d323b196465ebf60fdfc57461fcf6288e560695af571926` |
| `out_pb/x_opt125_gpt2_comm_Claude_s0.pt` | `b650172ab65048ccf85a2749ad0b7120628d5e44edfe140c8cec1b3415532d9f` |
| `out_pb/x_opt125_gpt2_comm_Claude_s1.pt` | `790ac76a674fd278a9aa917385ec75c4c4e1c22ac96fde67a081fa1da9fdf9c1` |
| `out_pb/x_opt125_gpt2_comm_real_s0.pt` | `ac9bc49073843ad2901fe62f0ec5ae9bbd9a9e2eef9151fcadf892622d41706e` |
| `out_pb/x_opt125_gpt2_comm_real_s1.pt` | `95a999fd89efdb00532324a6a5876a0d4ef03cd04df7dd483d15b589f2a65220` |
| `out_pb/x_opt125_gpt2_decomp.json` | `36e9409041a6037f412e1f24225da24939945e0032e8f5a438cc93eed582b196` |
| `out_pb/x_opt125_gpt2_meta.json` | `41a70ee8303a4d71cdc3d275a471c2aacfc6661728bffb9492598540b0897304` |
| `out_pb/x_opt125_gpt2_swap.npz` | `81e4626347fd83f355c8d3bf0e2535aeb9e2d18a273599be844558a1ed9f17b2` |
| `out_pb/x_pythia160_gpt2.stdout` | `67a8c2032cfa49a68908d8c068bd7358445db2b4d13dc8f0f5d037e7b9e63354` |
| `out_pb/x_pythia160_smol135.stdout` | `707b3c17f5e006c90f7fce5ed53efb63aaef294663768ac5a6d55b2840d0bf19` |
| `out_pb/x_skip.txt` | `03f392acb371abbaf381af228573d2ea2dedb3f0698081876e07e7b18d4f44be` |
| `out_pythia/bridges.pt` | `fe4e69c57caee7a20e56871259975b8a8dbb9ad33bf0ffcec970a29aa52a0fa8` |
| `out_pythia/gen_variants.sh` | `be0e8a42e151b6800a9133097ada65b8c3a0da6e94fd5884969625ecbe3f6caf` |
| `out_pythia/probes/all.txt` | `b53c4dc198a75d20a9ab1166fff9f40456582acd6fd3de0c528d319139d0ade3` |
| `out_pythia/probes/all3x.txt` | `13405650ff019ae06369e9982c18c58292c5c1b97a926bfc2a30a0713b3a75c7` |
| `out_pythia/probes/real_matched.txt` | `fef6d68bd20c529770b3afea0ca1a7d8bde28e691cf67199d0b94d50a9704567` |
| `out_pythia/probes/w0.txt` | `29382e4ff196e7e3faf95b6ca62ae84363fe4c21c84b2b4f719c937887e95fcf` |
| `out_pythia/probes/w1.txt` | `45633fa58f2a6c6e8d9f17bbb2e735e8a15daa37558391f61e848dd0e56cbacb` |
| `out_pythia/probes/w10.txt` | `801b1ed1034855056aade49e8c2fa9769b1f4406a1da55370e7ed78615141fbe` |
| `out_pythia/probes/w11.txt` | `d12044e832f4d08d974c41007ef00a399b0ae71c919da6950ff2b55f257054dc` |
| `out_pythia/probes/w12.txt` | `0539f3a35b0e5feec50b5463447cf5f5937a88b6ca737c0b88fd2f58e187a368` |
| `out_pythia/probes/w13.txt` | `a250f306664c2d8549744590fb11b0a3400e7390ed9167624cb11173ca0369db` |
| `out_pythia/probes/w14.txt` | `0aab8f2779f2d943c3290a6c0138985a9f9f53c54e01ff72055127990f75b888` |
| `out_pythia/probes/w15.txt` | `355de06e9d83f38b0b5fc6867b8f88de6cc478a1971fb1d0c486bb81d3b76ebf` |
| `out_pythia/probes/w16.txt` | `553653e4182c0c4540b94618f450e04c62fe9a5265c64fa19b9692739810d141` |
| `out_pythia/probes/w17.txt` | `40656b7be234bf850c9fdb9733b1db6b999acd6540d98b62688f06d7ab7efffd` |
| `out_pythia/probes/w18.txt` | `4f8915a5067955450e8621817219f5aa4e038ba4010f548ce919208238eb01e0` |
| `out_pythia/probes/w19.txt` | `4f98b6191302a3f4750f1b9e98485180841f6b31343488974d4ea12473d84851` |
| `out_pythia/probes/w2.txt` | `73406a43cb92fffc3ac5e359a0d5f08fc4b386d60e0b515bba955fcc608393a6` |
| `out_pythia/probes/w20.txt` | `4303a86986e7283ee5b0057cc22bd19d7a0e2e559796ccf9499f8be05427b0b4` |
| `out_pythia/probes/w21.txt` | `3a3916e18133cedaa5c0720fb6611e27d5538155f277a8fffb50482b8a3df75b` |
| `out_pythia/probes/w22.txt` | `0c13e29cdad47889aadd5985d71419dcb9191bf595bed9e7a092d9def110a8c8` |
| `out_pythia/probes/w23.txt` | `75118e752270aafd6c71fca33fc0f0e0646b47453f81a6a7be7b05bd24b0116b` |
| `out_pythia/probes/w24.txt` | `ef566e62053d1865b7ede5d96c24302be12bdc42985264747e98f2b3ac9046c7` |
| `out_pythia/probes/w25.txt` | `08385361a1068ac968b503363f950a59d678325cc70435549b3668b350864c90` |
| `out_pythia/probes/w26.txt` | `b70eece36227e8266ed061d4928845eb32e71edf431529d494ccfc9ec60917bd` |
| `out_pythia/probes/w27.txt` | `c981ca396b556b4aebe7b08dcc0c5020c21763385abe2aaee7b0d7303a115fb7` |
| `out_pythia/probes/w28.txt` | `d252d69903288293f942795d516543b352afe50cf02cd65e61d43a6468c350d5` |
| `out_pythia/probes/w29.txt` | `04225f6642aa07c5ef2c0364dc7b2322e2bb5bdf4d94474c39b74077636aa38e` |
| `out_pythia/probes/w3.txt` | `8a437f403f365016b2339814a7119f23e62e3920715a66968a6a95acea8c07a9` |
| `out_pythia/probes/w30.txt` | `8b53e0b87cc7abaedd0a614d4b843e8632d72cb302aab6c16cab224c2d450372` |
| `out_pythia/probes/w31.txt` | `eaf554a86e00310efb7994fdc7b2582d984e4b4db8e6b00baacaf858ae38c135` |
| `out_pythia/probes/w32.txt` | `8e4c7c1a75b8705e6c81cb639213f47e155988b38d636f5197e509b2f6301400` |
| `out_pythia/probes/w33.txt` | `e39060f1a81bd0eaf4ab8697eb9ef8708c8c7f4bf2ba01ddba69bb80c4538ff5` |
| `out_pythia/probes/w34.txt` | `78bf089117edbdee5bf56e4e823861c63862d2b1552a993711d5c7cab3f9deaa` |
| `out_pythia/probes/w35.txt` | `2b567f7055414af99c045817e7739fb9c9837f0b0830eecccc7c24a670330323` |
| `out_pythia/probes/w36.txt` | `48d65eb25197599873d8f074e19aec0f464ff6daacc7e08ce14c7344a26e0c64` |
| `out_pythia/probes/w37.txt` | `9bdfaccce751a8b95ae3e3c5dc816944339c68c3bdeb9484202a58e73f019b39` |
| `out_pythia/probes/w38.txt` | `5336ff1eba84c597356c055a61d622bf0964215a74e2eec8335d04f1a40c8501` |
| `out_pythia/probes/w39.txt` | `ef1c813d2eab564b18d162770d550026380dec0c02f0a89f40de446299fe4537` |
| `out_pythia/probes/w4.txt` | `cdce649528ec6ccfc6ca9014c6bc7209f2c1b5b4a3959c23f284b9f2feac238f` |
| `out_pythia/probes/w40.txt` | `3973005537155185ae60885a5b3c944cb6b69bd286ebc2130f0120a32d3a18bf` |
| `out_pythia/probes/w41.txt` | `b1a6360bb9446d6b692f346e2afe1700204fcc994eb28e9a3cf2c86eb0713dcf` |
| `out_pythia/probes/w42.txt` | `345861fb9d2c9ce49628833b2463ef443bec8b8b20117a505edd98535d7ec311` |
| `out_pythia/probes/w43.txt` | `110c9caf1d716e918cb9505ec0a337d7e0f0e2b94b9d176d16a5a1aa20bb30d6` |
| `out_pythia/probes/w44.txt` | `30b120653170629e1a360ae0c15734a66185cad6325ac6267d5198de49e74be9` |
| `out_pythia/probes/w45.txt` | `e817023f78c34e70c0237ff46574bdaca9c8ec9e96f91ac0568fb934d0398a6f` |
| `out_pythia/probes/w46.txt` | `bd93a569805b22e5e95e758cc07fecb8005b642ef084c56ebdf74067b9aaa9da` |
| `out_pythia/probes/w47.txt` | `e6f1f994d0923636e14e8c2798a7dc879a84d6f93367726f483de52f5ce91ca4` |
| `out_pythia/probes/w48.txt` | `c0c054ed8e27324fe0e56bf8caaf7b9731d3e8b5f6e29486253f4aff0bd79107` |
| `out_pythia/probes/w49.txt` | `9a93e224128ab6d30c2ae1a254bf80f93b850c222730bfa5b06d975567873963` |
| `out_pythia/probes/w5.txt` | `30e8ccc0190fea02f7dfb67cfc79f1bc49113c0468d36890b3eaaa33490ec70f` |
| `out_pythia/probes/w50.txt` | `ded38d079f5902a134f1fce297fcb2a9051387a664f596f5a9449ca509bc046e` |
| `out_pythia/probes/w51.txt` | `44b158e7e28e67fd14f959209452189783325d670219e618216a93682ea8bd2d` |
| `out_pythia/probes/w52.txt` | `e08ab2e723f537d0aeb50c9033aee6e21c294c973fb535a370ad0c4d0fb8d580` |
| `out_pythia/probes/w53.txt` | `ee33250b7e2d390845a39510cec110186f96d1d10dc94499828f7e4583e81213` |
| `out_pythia/probes/w54.txt` | `2a4e35c173d6871e42e7307f5ff3e3028fc5f6789b14d4a0ed3e8f4edcf4041c` |
| `out_pythia/probes/w55.txt` | `c04a4d774973fb4493fee66c75f8969a6ca27998651738aa4f131a7ce3a2541f` |
| `out_pythia/probes/w56.txt` | `75d3f57403e665218625e963675bf2129572d2f4fa51962db6a56e0c742f27aa` |
| `out_pythia/probes/w57.txt` | `7140e8fb8f9094ca824ab011553739644c817c3becb31916b22189acc8766bef` |
| `out_pythia/probes/w58.txt` | `c7588c3c9ca58b91e12ac3699109018ec555455f81fa4460aa34e893c97a576a` |
| `out_pythia/probes/w59.txt` | `d4559e3010a4c80a944f0e5fae65133ea40ebf921b6f20d47ed524473c900db2` |
| `out_pythia/probes/w6.txt` | `a408835b2217687b9566026aae98ab28e95e6db7581ac1cb7e8ff07b76de7685` |
| `out_pythia/probes/w60.txt` | `2fba846b8a2128e382519890272fa9c0d0e2cc8bf59d95d8155f2a82180f672d` |
| `out_pythia/probes/w61.txt` | `8facb4f42843f7d549300899ec494a8bff8680ac0ddba73dbf4b3412a637b5cc` |
| `out_pythia/probes/w62.txt` | `6baa947399689d622162db3fe977593e6967351bc9e10a680b5a1944e7e69524` |
| `out_pythia/probes/w63.txt` | `335ae528db0164271604191641a7a9a1c7a79248b52ded4999a91f2c318d1d2e` |
| `out_pythia/probes/w64.txt` | `b81b88042b4cf3eb84f24d8119e8a123f825c70c8921b0ccdadf2fdd6e834baa` |
| `out_pythia/probes/w65.txt` | `b892dc545a99ca57087a0a5532009348ddaf6a6cd476bfff8d4231cbbf6ad89c` |
| `out_pythia/probes/w7.txt` | `39c33e77fa39c398a9f1cc82c5e219f0609db9133f0777fac443a0d2867d89a7` |
| `out_pythia/probes_fiction/f0.txt` | `9361aeddfe77c7a11c5d56a68155f698de973f5712359352d999a56f427fa139` |
| `out_pythia/probes_fiction/f1.txt` | `e4e13608e7a0dd1c3167e26488ab2b09ea97b3955d55040b0f7491895dcc0e48` |
| `out_pythia/probes_fiction/f10.txt` | `0aa1a1731ea70e60b861eec7f4ae5085e2fafb94534f35d0cddbbf4c74b0ab0f` |
| `out_pythia/probes_fiction/f11.txt` | `a806750f6fadda257e58d41e30cc6f999178a08bc9d36255ee2e507839a8ed70` |
| `out_pythia/probes_fiction/f12.txt` | `5493b6f14c56d2d98c550068165bc7c69fe80e2b86e72de94406965446d1574a` |
| `out_pythia/probes_fiction/f13.txt` | `c8f3b21845fc348f59bee3c5f0291a04af20baac3bc321a664df5d8fbbadb519` |
| `out_pythia/probes_fiction/f14.txt` | `950987bc87ed3bc905027939f1264a6c58224dcc691506b4f4ff017a73d574bf` |
| `out_pythia/probes_fiction/f15.txt` | `20e1dc82a9d30ff0f88f41a71f132ad7be87f1ece3912cfb94079e208f13df70` |
| `out_pythia/probes_fiction/f16.txt` | `bc40924342f8bed32d97adb91231d1d31419be77f232ef26cda4c85a738744e5` |
| `out_pythia/probes_fiction/f17.txt` | `5863b95b935bfe6850ea0af42cb9277908975160fdca2403b0e26acc1e466c51` |
| `out_pythia/probes_fiction/f18.txt` | `4fae7540a38b8ae66258fca0174736ed2d1367fe60806d243c82b0343feb2851` |
| `out_pythia/probes_fiction/f19.txt` | `8c38bb863390752ce4fdd16778d7a991e6274a37e9cd9e2b30adc481d4c59f89` |
| `out_pythia/probes_fiction/f2.txt` | `4896901b494c7702b150c9e1e8d66aa0319674f7e5a81b223982ef48b9b994b6` |
| `out_pythia/probes_fiction/f20.txt` | `16e108f4c0554c02b0dcc6da17e6bb5cbb10cc66c0cd6994ed2653e517fce91f` |
| `out_pythia/probes_fiction/f21.txt` | `af211b04502a0650d794eaa210fd8f63b9551bc91b4028da8679ee4ebbbed5fe` |
| `out_pythia/probes_fiction/f22.txt` | `d7a17f7ca2b087a7afa901ce4ce2d89d7337421783329cb5332e4ce7179f6606` |
| `out_pythia/probes_fiction/f23.txt` | `67dab07914b88d059b165909acbf005e712146f05d4dc1bb27dbf312530a0f61` |
| `out_pythia/probes_fiction/f3.txt` | `77d10babfe8090a1a522dd002bdc15a0e077d4c297eb0068ecd37f2523c66f15` |
| `out_pythia/probes_fiction/f4.txt` | `37f8b972a52b6b2a5a031c00adfb26c8e935badab22b442116f190218b4939a8` |
| `out_pythia/probes_fiction/f5.txt` | `7507475f6110d02a9e93b140a2770da97328448fb30a8249d686f9165c884e3c` |
| `out_pythia/probes_fiction/f6.txt` | `c83600ba8c1de630b1ae4844f56ae9bd76726c03d9eb60410b9813cce3f70bf1` |
| `out_pythia/probes_fiction/f7.txt` | `b7d3504d3b90bd7e0da10c0d37abef9eefe1bc978a16f69bb0292710f9589de7` |
| `out_pythia/probes_fiction/f8.txt` | `6127b1fcbac2a59947965ea6b1d9724d78d4892087877db2ebe361161dbe5b07` |
| `out_pythia/probes_fiction/f9.txt` | `2ccb8ca2cbee2bfb5cdb95a6dc1d8ed34e7dc99f8c7daa350b5b1ea6ac8837fa` |
| `out_pythia/probes_fiction/topics.txt` | `bd639c50ed7d4b70582fc2cf0f64b8fa918e736fcc5377d292545c8d831f3f6b` |
| `out_pythia/probes_gpt/g0.txt` | `4dd229131a3852fc0ba26318480141242a07fb67c2dd920aab6f113a4848743c` |
| `out_pythia/probes_gpt/g1.txt` | `99affffcd734a6f2cb2820ea4f7fef4bacc53ed7159516417160d79f9c04a07c` |
| `out_pythia/probes_gpt/g10.txt` | `19b1225723de328ef3e87749f930e79788af0c1680a32451756c7e38e272ed1d` |
| `out_pythia/probes_gpt/g11.txt` | `58d1fc21f50b33312a24d5c9631750c747320706c1e0da22dd78036c3d29bb9b` |
| `out_pythia/probes_gpt/g12.txt` | `648c5257b443786d387ec0b6412a4a8939545eb83372b3fe11bafc5506c0dfc1` |
| `out_pythia/probes_gpt/g13.txt` | `004ddecec614f54299c814c536bb74acb220fe6239c6be2ef1c6aa6829bbc107` |
| `out_pythia/probes_gpt/g14.txt` | `3b50b9960471ea927b0990796828cd4c3b24f9d6684de0eeba64ba414a38f410` |
| `out_pythia/probes_gpt/g15.txt` | `244307e34305497e3c62573b4b7f9ee348313eb342914ea4c6f3c39371aa2313` |
| `out_pythia/probes_gpt/g16.txt` | `b115e83707a182adbd288032bf668473a81987d42e1a9fcdccdf60a73961352a` |
| `out_pythia/probes_gpt/g17.txt` | `04e686ee9ec5653467817b20c5896a7c649904308ed8c7f025dfde5dd9b3295c` |
| `out_pythia/probes_gpt/g18.txt` | `ba1f44c77174c9ed6e5af07656dfcad659f4f75feb8fc764de967f133ecc20e6` |
| `out_pythia/probes_gpt/g19.txt` | `f37c07ab9874ce1ef4393f27d99a23191ee9181f40ff6011639d566cca10ce95` |
| `out_pythia/probes_gpt/g2.txt` | `457210bc31d58d6850c068683e9f20cedb6dda57d5d58725ad0012cb0dc7563f` |
| `out_pythia/probes_gpt/g20.txt` | `cd6163c6ae9c5b77cff6ccaba75852f7cfa761bea5f5a23adc6e4349840c94ac` |
| `out_pythia/probes_gpt/g21.txt` | `a1fb40876b2eb1dd302820546260c6e486cd971ef4f9cb663d0a7928bfdf61b4` |
| `out_pythia/probes_gpt/g22.txt` | `d8f94f7143defae69428faff5dfd07f8fa3a2ae7762ed11f735bfb79f956b0ad` |
| `out_pythia/probes_gpt/g23.txt` | `546af10d4087ac7fd6eb38656c8709ce69e40d3fd6777a440e7548c43aaaf820` |
| `out_pythia/probes_gpt/g3.txt` | `a9c9581340eb516957f4f1e75af61402c6c737d912bd6bd236a91c587dbf594c` |
| `out_pythia/probes_gpt/g4.txt` | `b2c46a6e098fefaea58bc5a646e12ce464129f6c615bc0554bdc428c8c23b00e` |
| `out_pythia/probes_gpt/g5.txt` | `e87162f3a2c5ea8ee80563dcce7a21f7f8f5edf7e087eb14dd2220eb6d1a27e9` |
| `out_pythia/probes_gpt/g6.txt` | `c62cdd3e52cbdead675ae81701fa8e0a54ee01bad6f670ed36aa7cab4aabd120` |
| `out_pythia/probes_gpt/g7.txt` | `77b38cfec4a8483bf793bb8739971bf9643ea1a1c3142b5fc2517fc05df0ffe7` |
| `out_pythia/probes_gpt/g8.txt` | `f57fa24e0ae87f5131f3e8b29019b61be95f30194cda0e5d5a66cf289353975d` |
| `out_pythia/probes_gpt/g9.txt` | `2a3b0aa1d3c2f94e3a299adb76298aea969d1ee1f44e704554dbd9d7eba2321e` |
| `out_pythia/probes_gpt/topics.txt` | `bd639c50ed7d4b70582fc2cf0f64b8fa918e736fcc5377d292545c8d831f3f6b` |
| `out_pythia/regen_fiction.sh` | `7564f47cfe6419b5d3c798ced4073999a0e2f27ed04cdb1df53754c6580d45c4` |
| `out_pythia/results.json` | `26ca30a84d4225f65df89f1918e290fd937119b803216f7083adc45e97518e24` |
| `out_pythia/seeds.json` | `4640c2d5ff060f47a96157507816a471087ecd19d25be1690dd41f5eba3ca864` |
| `out_pythia/seeds.log` | `485883fecb45b5c2a732f5c9d4520d9ada61bec958b9d7d7786a0f4ca71577cb` |
| `out_pythia/seeds_per_window.npz` | `d8989762b636ca2c77d08bacd8803c682fd88f4421a425d20891a96fe9dd0495` |
| `out_lm/A.npz` | `9f35904b3683ca7dd9355c3ed893f9afddd37efa89c24c3b31d454113554ec40` |
| `out_lm/A.safetensors` | `2a4c1f96182c11a43ce4a8d7908ae0c63bf2b83d2c781d7c372d5d23e603a03b` |
| `out_lm/B.npz` | `afb2d378dfe4c479b8956cf7cf70c161afff736b6ee7961768c5708598026910` |
| `out_lm/B.safetensors` | `f8dcdc62847989b3ab5b6941516eb1738dccb8fca7d22e7d910ddc7f029edefd` |
| `out_lm/baselines.json` | `2a34a58bd8f95dff2fa5c6adbf0d7445f9823e9f1d9ef1af2e58b840afd835c0` |
| `out_lm/bridge_embed.npz` | `3b49ce3c83f6c10fa6ea4e80f5d75af7a8d6fcecacb2a785b19afca0bf733522` |
| `out_lm/bridge_task-loss fit on Claude-written probe text (no corpus).npz` | `20f44ad77c0f708e0437c6bddf810b633e9f580219f7d2d82289818ac68f2d02` |
| `out_lm/bridge_task-loss fit on real text (upper bound).npz` | `ac67fd7dabeef964cccdd3c62ed97262e827e0bf3b1ebcaf77c27a39c48ab9d6` |
| `out_lm/bridge_task-loss fit on shuffled probe characters (control).npz` | `71faf26f9992ed3316cba905e3f88fbeb74c288650abf682fee080c5b39feed3` |
| `out_lm/bridge_task-loss fit on text sampled from A and B themselves (no corpus, no LLM).npz` | `ccfc1ad9ad07d075f61182879e78f33f48ea0ddff2bd1f01624fce909b6b004b` |
| `out_lm/probe_overlap_audit.json` | `c89da93bd6e7fa3bdab22360c6643e46e734054350103845883b79e5bedde64c` |
| `out_lm/probes/all.txt` | `26c67ef298824850515e5a97d639cccb734b5cfab783cda9106574c04e9bfbe4` |
| `out_lm/probes/p0.txt` | `2604b8cc1a7b508255c596bee8b6f7c94ef885e32d838c4a55549f50a449cf63` |
| `out_lm/probes/p1.txt` | `b37d70a7395348f34f5c4a94a5f4ad26f2359c883811570c58ea7a2ede7afa18` |
| `out_lm/probes/p2.txt` | `c2d3011b5bc345fc6446886c4d770e0812fe711e06b1737181ee51ff162e7d22` |
| `out_lm/probes/p3.txt` | `3b61fe8463905b7cd914c0b488ee0477e036181b432d60483cd02b55478216db` |
| `out_lm/probes/selfsample.txt` | `761bb315ab72af1036361e6fa921129258e2b42487bcfdb2728469d5e2745fa1` |
| `out_lm/probes/shuffled.txt` | `21cec3798017e96af8f45cf09af7a78a2037e8f2a5b66a825522d69e8b3fc8d2` |
| `out_lm/seeds.json` | `8992aa4fb056f86679bb4365e8164159adb4aacb416e181ed7e615dd5fd903b2` |
| `out_lm/seeds.log` | `1074e5a79fa61ad395c609d7c931e372b79104b94d853477b193b42eb0c37016` |
| `out/baselines.json` | `6aae77fe7e0f71c7d9bc45912c548b84da034b02817b4ac502ca60f2ab80290b` |
| `out/hetero_A.npz` | `f573657ead7f7934099775b9c6eb7e4e3d5769bd85e54946dc8f6d3fb69bb2e5` |
| `out/hetero_B.npz` | `64539192f1c246067b5e9d559604c2de09f4226f0a3f3b91a30754a3c6e56197` |
| `out/hetero_bridge_cosine-match.npz` | `650fb83afee272b449eebd059ee039325954358374ec90ed8d3da76627b32629` |
| `out/hetero_bridge_weight-LSQ.npz` | `31161dee75720653e57434dc737a2ee8af96b052b99b2ec87d9495d2a1fb2601` |
| `out/llm/harness_hetero_0.log` | `e9b5b91dd66fe308b822fcb41606ad46e217c2ffbc4c8b6affc81143c992c1f0` |
| `out/llm/harness_hetero_1.log` | `6e56474f0076bf257b865b8cfe37b09c8d92835a40d103cb4673ee337b3daa60` |
| `out/llm/harness_hetero_9.log` | `98e34543dea86e23b57121bdf0d8bc82c24401ee5108c068f7c235fd61c63ea1` |
| `out/llm/harness_lm_0.log` | `33b775a3e94fec271ba69741d1b78807a341b9356eb0eed8be751a5e4f83ec4b` |
| `out/llm/harness_lm_1.log` | `60bc6119321f0a4ffc7df2c5070b2d62f7f93ce61d86f2b13ff17aac03ec46f3` |
| `out/llm/harness_same_0.log` | `3169bbe1a7dd30d1a359cb668516512a48ae8cad6c2088697ce71c512cc49d2e` |
| `out/llm/harness_same_1.log` | `ac4b36e955414c7cb206f81d5edee27efee0248c25a5b1d9ba8a9c3922d6c76f` |
| `out/llm/harness_same_9.log` | `09a7611a7fc80ae91c9527d25d253cd79dc522430357cc3e631a90efec6b4e49` |
| `out/llm/harness_width_0.log` | `8ca1c92db7ce6af684c4c6bf76b1c9e217a2fd60d8f28ddda172a9d64a8360f6` |
| `out/llm/harness_width_1.log` | `6a5c4d1ea5f85c3ae061d1b368f86110e17b81dbf38fba634aa18790270bdecd` |
| `out/llm/harness_width_9.log` | `c0241df14ce06751146ca336f202c5efdce904f27d95223717255d959091c6fb` |
| `out/llm/hetero_sandbox_r0.json` | `21f2533ae8cfd6eaea81ebf6e260a6a4a3562b062d42f2236af40b86e7047f5a` |
| `out/llm/hetero_sandbox_r0.jsonl` | `5fb1fa3f87a305a5d8801ea5eed13342f0a21c17fcbbf82342ae6c6122d183b6` |
| `out/llm/hetero_sandbox_r0_bridge.npz` | `c4671084d25b3abb9a0ffd6bbe0132fe94a8c246a297bc1be03cd5c5f990f73c` |
| `out/llm/hetero_sandbox_r1.json` | `a4313f9eb670a81cdb27156040600152c8edb8d8b27b76ec2e2f56a40c5c2706` |
| `out/llm/hetero_sandbox_r1.jsonl` | `ee47efcc6ebc79d56375c05b01a89c65276888d0cca97b8a945b95137396b967` |
| `out/llm/hetero_sandbox_r1_bridge.npz` | `a6f6a1e0530827bf9b6c1bf7ccfafeb2b99a50ab7cda116e8163ebec4a9c1ac9` |
| `out/llm/hetero_sandbox_r9.json` | `0b45e26db8bbbe78b991eb41b307f7cba316a12f980355d52c136f0019855e85` |
| `out/llm/hetero_sandbox_r9.jsonl` | `f52b6c4aae561a64ae41f78e9df961a3eee9e07ce8e8a4bab5d22a10987c046f` |
| `out/llm/hetero_sandbox_r9_bridge.npz` | `c3aad99c893c72923e067856b86460dbb8198a390d3c00be9b960c0b44a9b3dd` |
| `out/llm/hetero_toolfree-high_r0.err` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `out/llm/hetero_toolfree-high_r0.json` | `ee0c5b71cd3fca5f24606d19b948b3fe4aee79913d04a5dcdb5613c7cd242214` |
| `out/llm/hetero_toolfree-high_r1.err` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `out/llm/hetero_toolfree-high_r1.json` | `b6877bdeeede4f2074a58c98761512af652f818f68f150f2e6821fd224f50028` |
| `out/llm/hetero_toolfree-high_r2.err` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `out/llm/hetero_toolfree-high_r2.json` | `9d7a79063c4a9c2b01725e205d27e274cfbf19203762c50a7773f3e40ed53d46` |
| `out/llm/lm_sandbox_r0.json` | `dfc3a23f8b0b9d11fca18e7c45d7b301c00c826e7a152578f8aa54353a3bef15` |
| `out/llm/lm_sandbox_r0.jsonl` | `cb47660c486b9c2800e1958b13e77e0a923fcf1d7c51edd73090810aa6e6ed03` |
| `out/llm/lm_sandbox_r0_bridge.npz` | `cc2d9eb0eef857e01dd1e374271cda40c1932526c71f7278a88544fd8c9db40c` |
| `out/llm/lm_sandbox_r1.json` | `8bf898d442cbd4e4a325a737ee1939c34d27f25e487af3d102fe2aa2209cc355` |
| `out/llm/lm_sandbox_r1.jsonl` | `baedca25980e9dbbab46e9df03ac69cc564a4fedaa78b7150b9c267f6895d456` |
| `out/llm/lm_sandbox_r1_bridge.npz` | `130310df39752960584f73bfa35e32a4fba54765042ac858fbb102aebee15427` |
| `out/llm/same_sandbox_r0.json` | `21ec6dd9412971de8c7c00f2ed386b8f6f33d5969c8603dad64066b9c7186c2d` |
| `out/llm/same_sandbox_r0.jsonl` | `fbc4375887d8712e0cbf8a19902f31f5e3c52827848f0c4641c3c2eb53a0c03c` |
| `out/llm/same_sandbox_r0_bridge.npz` | `3dc67f88c29e595b98713b09adf66670acd3290c9cc8204213fb86d929ae001c` |
| `out/llm/same_sandbox_r1.json` | `a824258fa4bb33ae765c81fb13d6c2ecd0de752c36c10740cf89a621b140c52c` |
| `out/llm/same_sandbox_r1.jsonl` | `b58fb1461f9943d5a0503ea30aa7aa290592e9c80935b00a96e80d104028d448` |
| `out/llm/same_sandbox_r1_bridge.npz` | `5e36bddfdcf102e5bcd18998e0bcff88334add0d18b5c13e293f4ecf595b251c` |
| `out/llm/same_sandbox_r9.json` | `b4cca55c5b064a576f567a0d9f2e8123c5f4ced4c2c9d612e4f281b70f37bf0f` |
| `out/llm/same_sandbox_r9.jsonl` | `15fe43d01d36f2370798f07e590563f14af7b03d4c7dfc9247a43fdbf898d083` |
| `out/llm/same_sandbox_r9_bridge.npz` | `c2b5c662e41e0e4c524c7bd1247855c906380decec6024884d759e3c05d0d62f` |
| `out/llm/same_toolfree-high_r0.err` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `out/llm/same_toolfree-high_r0.json` | `09f333b48a89fc523df8bdaf63da43e8ef8b7be9cb01c9ef73732ce6fb211791` |
| `out/llm/same_toolfree-high_r1.err` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `out/llm/same_toolfree-high_r1.json` | `533fc4a4b408c03f4cc860493cd9a076f0d1a885e26f59f056211c8fcbca0352` |
| `out/llm/same_toolfree-high_r2.err` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `out/llm/same_toolfree-high_r2.json` | `1450aabdf88dadd6d1ae383a557eff4cbcd184a9647759893310655131657be9` |
| `out/llm/width_sandbox_r0.json` | `73a00852be30093aabd4ca4e11dd6e78ce58873086d7aa6abe9a95ff396928d2` |
| `out/llm/width_sandbox_r0.jsonl` | `162b8ae859ea7e7420c81ed68a054e763eaaffd93d19da2a78cb64140d4df5b2` |
| `out/llm/width_sandbox_r0_bridge.npz` | `db45db46c5a220ffb514f007fcc1c93c292cd975b8d96bf5999e3a01849965ad` |
| `out/llm/width_sandbox_r1.json` | `e010e983c97f6a443731e571a5edceeb0fc8dd4bf31fca19c722d611758475ce` |
| `out/llm/width_sandbox_r1.jsonl` | `ceb422e2189b7c6ec76c26c710f41687e0f14d4528c084e27066ff057c5b6137` |
| `out/llm/width_sandbox_r1_bridge.npz` | `114ac63a53301ad934081104cbe99d3221f18de4741668ccc0e6de3028299602` |
| `out/llm/width_sandbox_r9.json` | `3ba93179dcc04ad35b52554a976eb27455872e3e0bde711a3289e5f2ea09686e` |
| `out/llm/width_sandbox_r9.jsonl` | `0552fc1192e1eae846a7b04f66f5d9e415f2541063a3270f906de10d571e1cc7` |
| `out/llm/width_sandbox_r9_bridge.npz` | `0a7641447e608b9b17b7685216f3fec6568c9dcbb7d7b1b9b91f0171ca901a9c` |
| `out/llm/width_toolfree-high_r0.err` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `out/llm/width_toolfree-high_r0.json` | `e902d90f39602464dd47c10555a95ec83ced190ef015ad8e3ecfdff024e63c62` |
| `out/llm/width_toolfree-high_r1.err` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `out/llm/width_toolfree-high_r1.json` | `81db42f1cb4c8d8375f460cbdef3ef0d10a94bfe3ed5f9923b99277650ab67ff` |
| `out/llm/width_toolfree-high_r2.err` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `out/llm/width_toolfree-high_r2.json` | `a5df7df5f97ec726ecedddd7325961f4d33a133bca99c3ec6ab780bbe1064503` |
| `out/llm_scores.json` | `de36add86ff57a065689be0cb00db380d02874f903e6b2b44668b18602baf030` |
| `out/noise_distill.json` | `361c7dad45a47959ff5ef8e76ea633c015a0c0e040f90192b14baa33f3aafd47` |
| `out/noise_probe.json` | `c2a7e85af5bce74d54a36cf049f6e444b1bac5e89973ed77499f99fd81d73b7a` |
| `out/prompt_hetero.txt` | `3473992b9e1230a5516f07af8eadd895a38035b43eaca5639ceefa9a0844477b` |
| `out/prompt_same.txt` | `c01696cfd71323ad63b4db26c67e21e76fdf1bc1fc1134ab99c3c6a069da0cd2` |
| `out/prompt_width.txt` | `1a2cd753b0d2f7fefb1bf45e94d954320fc6c9149b1325cf22e45730d54cd395` |
| `out/same_A.npz` | `d249e8ba814a3db351056b09108b1e0a1c5abaae9a51bd131d37d23e5c3b585b` |
| `out/same_B.npz` | `7ee116ebf019a3f0f9979f5227da91e1b36b60c434b800bee6c88cc34592e206` |
| `out/same_bridge_cosine-match.npz` | `c742db082a5b997764635f713250cc514665fd278d9c11e43cb0dc2aaf613b10` |
| `out/same_bridge_weight-LSQ.npz` | `62be07351e5b56fac964463332636b5143eb324a81cf5192863cde575f63d198` |
| `out/width_A.npz` | `242956e2c6fef8d0d6cf682429c38fa925728c4ba93a7d78afac633fe0896f60` |
| `out/width_B.npz` | `f7557f2d0d8d77dfb33f5e6ae44a6c80e0cbae8ce3577cdd04f976ba988589b8` |
| `out/width_bridge_cosine-match.npz` | `686eb85edca3ded73f12bab9dbe06d44c2f4e8e30bb44a36938509a40ace952d` |
| `out/width_bridge_weight-LSQ.npz` | `2ebe0173e470eb72b3e0b55af81468e0dc095a75ba3533f29dbe8cdb9672697c` |
| `../paper/fancyhdr.sty` | `b56ec4434b9f4607529a4b23dc68ad8d4b94f1f631c8cddaf7da78140d53a5ea` |
| `../paper/figs/audit_panel_alignment.py` | `43958e6b49d897f326ff64e73cbed0bf057e17c06533b5b57d3dededf8111e76` |
| `../paper/figs/fig_comm.alignment.json` | `ae00fb4c739433fcc18c4f0c9b1ed01b25c50899b8d3339f321648e7905574c1` |
| `../paper/figs/fig_comm.alignment.svg` | `589b16216c141325314a50af99b985bca39da24c022e040e4a99f3e9fd8ed90d` |
| `../paper/figs/fig_comm.collision-audit.json` | `905736df682179e6ee9553b338dc8761a5bb6fa3f1228f21f1175cba55c789a1` |
| `../paper/figs/fig_comm.collision-audit.pdf` | `eb73bb65c7cda00cd6385d5b1abe369b20381232ebc079b52499f4e3b28953d9` |
| `../paper/figs/fig_comm.pdf` | `72602a4855d8aecdf5721a820ecd37be71ad07808d5ca1731329065e9490c60a` |
| `../paper/figs/fig_comm.png` | `ae4816dd49e642ec0e8fd671f739b4105007908f84971a72f2df218acb61aaab` |
| `../paper/figs/fig_comm.qa.md` | `b534a482b040959c9392867c6c211d6a9bb725e4bf9bec3e0fbb77b430584222` |
| `../paper/figs/fig_comm.source.json` | `da341903953b5db0ebddc047c48c1bfdb712171a73ae3a50a1feddc23afd9af0` |
| `../paper/figs/fig_comm.svg` | `04ec0ad0fdb239c2ca7975fbd5eee4f410853ab7b072bdf6828ccf49990f26a3` |
| `../paper/figs/fig_comm.tiff` | `4b3384bb55f66fbef04bbd050baa632414e5be6707f3700b60001791e2e2963f` |
| `../paper/figs/make_fig_comm.py` | `342f365b7b6a2c5bc54a17a8430c686c2b1534e4ade5b76f0d59e929dfd1732a` |
| `../paper/figs/requirements-fig-lock.txt` | `119252489c872920ee91a3ff3c7468720e82f2ce04e2b26b5f576320491530c0` |
| `../paper/main.pdf` | `f449abde1b804e90d5e554fb96be7848c76cf3569447a73a1cfde449c94e62c4` |
| `../paper/main.tex` | `cde191d9c441cfb2b9dffb4d3c17964314170bcae536488224287c75bfb7dc4f` |
| `../paper/math_commands.tex` | `90473c4d0542070db244cea73ef962d6cddc5b2a746757e6a40ddf5fdfb90ba9` |
| `../paper/natbib.sty` | `88bc70c0e48461934cab5b2accef06b74a8b3ac45ad03ccd3f2a6b7e0d6d530d` |
| `../paper/placeholders.tex` | `6599bbed2b9146dd97d9d19c778ed206d75eebcb61b3aa151586f7983fab3302` |
| `../paper/refs.bib` | `2c4371e42b2ab6fedac00b9ce3e9b443e527bbfb982353291f1628e4fd05fa7d` |
| `../paper/sec_abstract_intro.tex` | `7d4944a778c8218d997902f0c485329a99c14b5155406fc0160666b2fc49c64b` |
| `../paper/sec_appendix.tex` | `2c0db7100d6e266eab8ae1db02783df949bd53218692b18c062349a38ca3cb80` |
| `../paper/sec_appendix_rest.tex` | `7bab0ec0ff860c1b36039bfba92d81dd5a2a16bd33b73b4baa70e183e7f073db` |
| `../paper/sec_appendix_small.tex` | `661b2f110699a0bd45223f9b2bf2ee554893beda8e274d53a80ba75107e20870` |
| `../paper/sec_bandwidth.tex` | `8a4ab69a3f19657f18eeaff7a8e59660bb534dcc3bd12518833049bec6a58f9b` |
| `../paper/sec_diag.tex` | `1b78520743efdb981a4c7175e503192b49772bdb10fe67d2b7e538547ab2b985` |
| `../paper/sec_discussion.tex` | `ce145eeec8776def92ddd16ec729e917cb1eb5f1d5e341ae8d6c4720b59971a8` |
| `../paper/sec_general.tex` | `d894e25cf68894d2aaa9555e623e617aa6f3dae233ba01c13abca4cf6ee7cda6` |
| `../paper/sec_related_method.tex` | `848d6140674c20108f8f90034e8d195b8a6e935a141076d43367fbaeca5c8eb0` |
| `../paper/sec_results.tex` | `55a6c71e66c0d3a3d5f157cd7b02a48bc45cb2760431c96a581b69e76db21bec` |
| `../paper/sec_results_rest.tex` | `97f14c8776db13866777f84727cca11cc1992aa3c2972de1b2af288760304712` |
| `../paper/tab_comm70.tex` | `ff5e5ab53a82c6e2c821c7e37d7e78dcd68bf42794bc2af36e2bd5c3e612d1a6` |
| `../paper/tab_general.tex` | `4c57941c05f9930d891dfbeb0f850f73d37bb3b134524ea8c25fe2ecfabe692b` |

## Superseded outputs (not used by the paper)

- `comm70.json/.npz` keys `... | mismatched message`: the first donor control (messages rolled within each batch of 16;
  83% of donors from the same article). Matched values in these files are still valid and identical to `comm70_swap`.
- `comm70_summary.json` (pb_analyze.py), `batchroll_comm70_extra.*`: analyses built on that control.
- `v2_comm70_swap.*`, `v2_comm70_decomp.json`: five random donors without whole-article restriction; superseded by v3.
- `farshift_comm70_extra.*`: per-position analysis with a deterministic half-shift donor; superseded.
- `textscale_aborted_batchroll.*`: an aborted text-scaling run with the first control.
- `unpaired_sender_compare.json`: sender comparison whose bootstrap drew seeds and donors independently for the two
  senders; superseded by the paired `sender_compare.json`.
- Legacy output keys: early outputs (`comm70.json/.npz`, `sweep70*`, `ceiling.txt`) label the receiver-state reference
  "oracle message ... (protocol ceiling)" or "oracle"; these are the same quantity, renamed in the paper.
- `sanity.*` uses the first control for its comm entries; only its hand-over identity check is cited.
- `variants_aborted*`, `textscale_aborted*`: runs stopped before completion (incomplete probe files / trimmed design).
- Fiction files f18, f20, f21, f23 were regenerated after interrupted calls (`regen_fiction.sh`); failed attempts were
  overwritten and are not archived. The GPT corpus was generated with `codex exec --sandbox read-only` (model not pinned,
  transcripts not kept).
- Cross-family self-sample probe text (pbx.py) was generated in memory and not saved; its filtering counts are therefore
  not available (real-text and LLM-text counts are in out_pb/general_audit.json).

