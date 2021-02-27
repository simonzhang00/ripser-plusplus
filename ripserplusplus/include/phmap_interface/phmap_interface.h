#pragma once

void phmap_put(int64_t, int64_t);//put the key,value pair into the hashmap

int64_t phmap_get_value(int64_t);//return -1 if key is not in the hash map, otherwise the corresponding value of the key

void phmap_clear();//clears the hashmap