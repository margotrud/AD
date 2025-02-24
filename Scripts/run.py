from adni_processor import ADNIProcessor

is_need_to_build_DBB = True
if __name__ == "__main__":
    if is_need_to_build_DBB:
        processor = ADNIProcessor()
        processor.run_pipeline()


