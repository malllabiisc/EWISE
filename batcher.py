def batcher(file_input, batch_size):
    import numpy as np
    import json

    data = json.load(open(file_input))

    list_of_dict = []

    for sents in data:
        ip_array = np.fromstring(sents['original'].strip(), dtype = int, sep= ' ')
        op_array = np.fromstring(sents['annotated'].strip(), dtype = int, sep= ' ')
        try:
            list_of_dict.append({
            'original': ip_array,
            'annotated': op_array,
            'offsets': sents['offsets'],
            'doc_offset': sents['doc_offset'],
            'stems': sents['stems'],
            'pos': sents['pos']
            })
        except Exception as e:
            print e

    buckets = [[w for w in list_of_dict if w['original'].shape[0] == num] for num in set(i['original'].shape[0] for i in list_of_dict)]
    
    final_ip_list = []
    final_op_list = []
    final_offset_list = []
    final_doc_offset_list = []
    final_stem_list = []
    final_pos_list = []
    for ele in buckets: # ele is a list of json
        temp_ele = [ele[i:i + batch_size] for i in xrange(0, len(ele), batch_size)]
        for elem in temp_ele:
            ip_arr2d = np.array([elems['original'] for elems in elem])
            op_arr2d = np.array([elems['annotated'] for elems in elem])
            final_ip_list.append(ip_arr2d)
            final_op_list.append(op_arr2d)
            temp_offset_list = []
            temp_doc_offset_list = []
            temp_stem_list = []
            temp_pos_list = []
            for elems in elem:
                temp_offset_list.append(elems['offsets'])
                temp_doc_offset_list.append(elems['doc_offset'])
                temp_stem_list.append(elems['stems'])
                temp_pos_list.append(elems['pos'])

            final_offset_list.append(temp_offset_list)
            final_doc_offset_list.append(temp_doc_offset_list)
            final_stem_list.append(temp_stem_list)
            final_pos_list.append(temp_pos_list)

    return zip(final_ip_list, final_op_list, final_offset_list, final_doc_offset_list, final_stem_list, final_pos_list)
