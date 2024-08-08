#ifndef EVB_TEXT_H
#define EVB_TEXT_H

#include <stdio.h>

#define MAX_WORD 5000
#define LEN_LINE 2000
#define MAX_KEY 1000
#define INC_BUF 10000

class Text
{
  public:
    
  Text(char *, FILE *);
  ~Text();
  
  int word[MAX_WORD];
  char* key[MAX_KEY];
  char* key_content[MAX_KEY];
  char* buf;
  
  int buf_size;
  int nch;
  int nword;
  int nkey;
  
  int ignore;
  int layer,fate;
  int level;

  void read_file(char*, FILE*, FILE *);
};


#endif
