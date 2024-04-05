import re 

def extract_text(
        input_file, 
        output_file, 
        chunk_size=1024, 
        total_bytes_to_read_in_mb=500, 
        encoding='utf-8'):
     
    with open(input_file, 'rb', encoding=encoding) as f_in: 
        with open(output_file, 'wb', encoding=encoding) as f_out: 
            total_bytes_to_read = total_bytes_to_read_in_mb * 1024 * 1024  
            bytes_read = 0
            while bytes_read < total_bytes_to_read: 
                chunk = f_in.read(chunk_size) 
                if not chunk:
                    break 
                f_out.write(chunk) 
                bytes_read += len(chunk)
            print("Extraction completed successfully.")
