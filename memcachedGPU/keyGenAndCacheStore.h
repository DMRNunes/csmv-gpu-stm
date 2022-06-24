#ifndef KEY_GEN_AND_CACHE_STORE
#define KEY_GEN_AND_CACHE_STORE
  
#ifdef __cplusplus
extern "C" {
#endif // C++

// used below, no need to call these
void zipf_setup(unsigned long nbItems, double param);
unsigned long zipf_gen();

void fill_array_with_items(int nbItems, unsigned long itemSpace, double param, void(*callback)(int idx, int total, long long item));

// returns 0 if file does not exist, returns -1 and exits if file exists 
int storeArray(const char *name, long nbItems, int sizeItem, void *values);

// returns -1 if file does not exist, returns 0 and loads if file exists 
int loadArray(const char *name, long nbItems, int sizeItem, void *values);

#ifdef __cplusplus
}
#endif // C++

  

#endif /* KEY_GEN_AND_CACHE_STORE */