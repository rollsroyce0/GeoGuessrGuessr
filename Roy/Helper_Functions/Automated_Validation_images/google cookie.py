with open('Roy/ML/Second_Level_ML/generate_coordinates.py', 'r+') as f:
        lines = f.readlines()
        # find the index of the line that says "    # New If-Else below here", "        #new test types added here" or "    # New arrrays below here", all separatedly
        insert_index_array = None
        insert_index_ifelse = None
        insert_index_testtypes = None
        for i, line in enumerate(lines):
            if line.strip() == "    # New If-Else below here":
                insert_index_ifelse = i + 1
                continue
            if line.strip() == "        #new test types added here":
                insert_index_testtypes = i + 1
                continue
            if line.strip() == "    # New arrrays below here":
                insert_index_array = i + 1
                continue
        if insert_index_array is not None:
            print(f'Inserting new array at line {insert_index_array}')
            lines.insert("TEST")
            
        if insert_index_ifelse is not None:
            print(f'Inserting new if-else at line {insert_index_ifelse}')
            lines.insert("TEST")
        if insert_index_testtypes is not None:
            print(f'Inserting new test type at line {insert_index_testtypes}')
            lines.insert("TEST")
        
        f.seek(0)
        f.writelines(lines)
            