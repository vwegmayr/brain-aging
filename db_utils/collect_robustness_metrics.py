from db_utils.robustness_records import per_run_table, summary_table, Record


RECORDS = [
    Record(0, "20180921-101852", 1),
    Record(0, "20180921-123619", 7),
]


if __name__ == "__main__":
    per_run_table(RECORDS)
    summary_table(RECORDS)
    
