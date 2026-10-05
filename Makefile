# Entry points for the Barca fine-tuning workflow.
#
#   make sync                     copy this repo to the GPU server
#   make train                    fine-tune there with the default recipe
#   make train RECIPE=configs/full_ft.yaml
#   make train RECIPE=configs/qlora.yaml NUM_GPUS=1
#   make inspect                  show dataset splits/lengths on the server
#   make pull checkpoint-1000     copy a checkpoint back here
#
# Defaults live below; override any of them in Makefile.local (gitignored) or
# per invocation. See Makefile.local.example.

-include Makefile.local

SSH_HOST   ?= 41.226.183.241
SSH_PORT   ?= 2224
SSH_USER   ?= kabidi
REMOTE_DIR ?= /home/kabidi/workspace/projects/barca
REMOTE_PY  ?= /opt/miniconda3/bin/python
DATA       ?= /home/kabidi/workspace/datasets/uniref50/barca

RECIPE     ?= configs/lora.yaml
NUM_GPUS   ?= 1
EXTRA_ARGS ?=

# Where `train` writes on the server, and where `pull` copies checkpoints to.
REMOTE_RUN_DIR ?= outputs/barca-0.8b-lora
LOCAL_RUN_DIR  ?= runs

SSH   := ssh -p $(SSH_PORT) -o BatchMode=yes $(SSH_USER)@$(SSH_HOST)
RSYNC := rsync -az

.DEFAULT_GOAL := help

.PHONY: help sync train finetune inspect pull

help:
	@echo "Barca workflow targets:"
	@echo "  make sync                 rsync this repo to $(SSH_USER)@$(SSH_HOST):$(REMOTE_DIR)"
	@echo "  make train                train on the server (RECIPE=$(RECIPE), NUM_GPUS=$(NUM_GPUS))"
	@echo "  make inspect              inspect DATA=$(DATA) on the server"
	@echo "  make pull <checkpoint>    copy $(REMOTE_RUN_DIR)/<checkpoint> into $(LOCAL_RUN_DIR)/"
	@echo ""
	@echo "Override defaults in Makefile.local (gitignored) or on the command line."

sync:
	$(RSYNC) \
		--exclude='__pycache__/' --exclude='*.py[cod]' --exclude='.pytest_cache/' \
		--exclude='.venv/' --exclude='outputs/' --exclude='runs/' --exclude='wandb/' \
		--exclude='Makefile.local' \
		-e "ssh -p $(SSH_PORT) -o BatchMode=yes" ./ \
		$(SSH_USER)@$(SSH_HOST):$(REMOTE_DIR)/

train:
	$(SSH) 'cd $(REMOTE_DIR) && PYTHON=$(REMOTE_PY) NUM_GPUS=$(NUM_GPUS) bash scripts/train.sh $(RECIPE) $(EXTRA_ARGS)'

# Alias kept for parity with the sibling project's workflow.
finetune: train

inspect:
	$(SSH) 'cd $(REMOTE_DIR) && PYTHONPATH=src $(REMOTE_PY) scripts/inspect_dataset.py $(DATA)'

# `make pull checkpoint-1000`. Make treats every word of the goal list as a
# target, so the checkpoint name is lifted out of MAKECMDGOALS here, and the
# no-op pattern rule below absorbs it before make can look for a rule of its own.
ifneq ($(filter pull,$(MAKECMDGOALS)),)
CHECKPOINT := $(filter-out pull,$(MAKECMDGOALS))
endif

pull:
	@test -n "$(CHECKPOINT)" || \
		{ echo "usage: make pull <checkpoint-dir>   e.g. make pull checkpoint-1000"; exit 1; }
	mkdir -p $(LOCAL_RUN_DIR)
	# optimizer.pt is the AdamW state; excluded to keep the copy small.
	rsync -avz --partial --exclude='optimizer.pt' \
		-e "ssh -p $(SSH_PORT) -o BatchMode=yes" \
		$(SSH_USER)@$(SSH_HOST):$(REMOTE_DIR)/$(REMOTE_RUN_DIR)/$(CHECKPOINT)/ \
		$(LOCAL_RUN_DIR)/$(CHECKPOINT)/

ifneq ($(filter pull,$(MAKECMDGOALS)),)
%:
	@:
endif
