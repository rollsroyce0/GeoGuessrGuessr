with open('Roy/ML/Second_Level_ML/generate_coordinates.py', 'r+') as f:
        lines = f.readlines()
        print(lines)
        
        
        # find the index of the line that says "    # New If-Else below here", "        #new test types added here" or "    # New arrays below here", all separatedly
        insert_index_array = None
        insert_index_ifelse = None
        insert_index_testtypes = None
        for i, line in enumerate(lines):
            print(f'Line {i}: {line.strip()}')
            if line.strip() == "# New arrays below here":
                insert_index_array = i + 1
                continue
            if line.strip() == "#new test types added here":
                insert_index_testtypes = i + 3
                continue
            if line.strip() == "# New If-Else below here":
                insert_index_ifelse = i + 2
                continue
            
        print("\nModifying generate_coordinates.py...")
        print(f'Found insert indices - Array: {insert_index_array}, If-Else: {insert_index_ifelse}, Test Types: {insert_index_testtypes}')
        if insert_index_array is not None:
            print(f'Inserting new array at line {insert_index_array}')
            lines.insert(insert_index_array, "TEST\n")
            
        if insert_index_ifelse is not None:
            print(f'Inserting new if-else at line {insert_index_ifelse}')
            lines.insert(insert_index_ifelse, "TEST\n")
            
        if insert_index_testtypes is not None:
            print(f'Inserting new test type at line {insert_index_testtypes}')
            lines.insert(insert_index_testtypes, "TEST\n")
        
        f.seek(0)
        f.truncate()
        f.writelines(lines)
            