FASTQ_PATH = config["fastq_path"]
REF_PATH = config["ref_path"]
THREADS_NUM = config["align_threads"]
OUTOUT_PATH = config["output"]

rule all:
    input:
        expand("{output_path}/merged.fastq.gz", output_path=OUTOUT_PATH),
        expand("{output_path}/align.sam", output_path=OUTOUT_PATH),
        expand("{output_path}/read_ids.txt", output_path=OUTOUT_PATH)

rule cat:
    input:
        FASTQ_PATH
    output:
        "{output_path}/merged.fastq.gz"
    shell:
        "cat {input}/*.fastq.gz > {output}"

rule alignment:
    input:
        "{output_path}/merged.fastq.gz"
    output:
        "{output_path}/align.sam"
    shell:
        "minimap2 -ax map-ont {REF_PATH} {input} -t {THREADS_NUM} > {output}"

rule getid:
    input:
        "{output_path}/align.sam"
    output:
        "{output_path}/read_ids.txt"
    shell:
        "samtools view -q 1 {input} | cut -f 1 | sort | uniq > {output}"