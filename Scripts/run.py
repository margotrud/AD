from adni_processor import ADNIProcessor

is_need_to_build_DBB = True
Longitudinal_analysis = False
if __name__ == "__main__":
    if is_need_to_build_DBB:
        processor = ADNIProcessor()
        processor.run_pipeline()
        if Longitudinal_analysis:
            processor.extract_baseline_vars()
        else:
        # Baseline analysis
            processor.filter_baseline_data()





