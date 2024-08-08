#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "EVB_text.h"

/* ------------------------------------------------------------------ */

Text::Text(char *fname, FILE * fp_cfg)
{
    nkey = nch = nword = 0;
    ignore = layer = 0;
    
    memset(key,0,sizeof(char)*MAX_KEY);
    memset(key_content,0,sizeof(char)*MAX_KEY);

    buf_size = INC_BUF;
    buf = (char*) malloc(sizeof(char)*buf_size);
    
    FILE *fp = fopen(fname,"r");
    if(!fp)
    {
        printf("ERROR: Cannot open file [%s].\n", fname);
        exit(1);
    }

    level = 0; // Current level of read_file

    read_file(fname,fp,fp_cfg);
    fclose(fp);

    /*****************************
    // key
    printf("\n");
    for(int i=0; i<nkey; i++)
    {
        printf("#define   %s\t",key[i]);
        if (key_content[i]) printf("%s",key_content[i]);
        printf("\n");
    }
    
    //for(int i=0; i<nword; i++) printf("%s\n",buf+word[i]);
    /*****************************/
}

Text::~Text()
{
  free(buf);
  for(int i=0; i<nkey; i++) free(key[i]);
  for(int i=0; i<nkey; i++) free(key_content[i]);
}

/* ------------------------------------------------------------------ */

void Text::read_file(char *fname, FILE *fin, FILE * fp_cfg)
{
  level++;

    int line_id = 0;
    char line[LEN_LINE+1];

    while(fgets(line,LEN_LINE,fin))
    {
        line_id ++;
        char* p = strtok(line," \t\n");
                
        //////////////////////////////////////////
        // blank or comment line
        //////////////////////////////////////////
        if(!p || *p==':') continue;
        
        //////////////////////////////////////////
        // ignoring flag on
        //////////////////////////////////////////
        if(ignore)
        {
            if(strcmp(p,"#ifdef")==0) fate++;
            else if(strcmp(p,"#else")==0 && fate==0) ignore = 0;
            else if(strcmp(p,"#endif")==0)
            {
                if(fate)
                {
                    fate--;
                    continue;
                }

                ignore = 0;
                layer--;
                if(layer<0)
                {
                    printf("ERROR: unmatched [#if] and[#endif] at [%s:%d].\n",fname,line_id);
                    exit(1);
                }
            }
            continue;
        }
        
        //////////////////////////////////////////
        // #include
        //////////////////////////////////////////
        
        if(strcmp(p,"#include")==0)
        {
            if(ignore) continue;
            
            p = strtok(NULL," \t\n");
            if(!p || *p==':')
            {
                printf("ERROR: incompleted command [#include] in [%s:%d].\n", fname,line_id);
                exit(1);
            }

	    if(*p=='\"') p++;
	    int _t = strlen(p)-1;
	    if(p[_t]=='\"') p[_t]=0;

            FILE *fp = fopen(p,"r");
            if(!fp)
            {
                printf("ERROR: Cannot open file [%s].\n", p);
                exit(1);
            }
            read_file(p,fp,fp_cfg);
            fclose(fp);
        }

        //////////////////////////////////////////
        // #define
        //////////////////////////////////////////
        else if(strcmp(p,"#define")==0)
        {
            p = strtok(NULL," \t\n");
            if(!p || *p==':')
            {
                printf("ERROR: incompleted command [#define] in [%s:%d].\n", fname,line_id);
                exit(1);
            }

            int ikey;
            for(ikey = 0; ikey<nkey; ikey++) if(strcmp(key[ikey],p)==0) break;
            
            int len1 = strlen(p);
            key[ikey] = (char*) malloc(sizeof(char)*(len1+1));
            strcpy(key[ikey],p);
            key[ikey][len1]=0;
            
            p = strtok(NULL," \t\n");
            if(p && *p!=':')
            {
                int len2 = strlen(p);
                key_content[ikey] = (char*) malloc(sizeof(char)*(len2+1));
                strcpy(key_content[ikey],p);
                key_content[ikey][len2]=0;
            }
            else key_content[nkey]=NULL;

            if(ikey==nkey) nkey++;
        }

        //////////////////////////////////////////
        // #ifdef
        //////////////////////////////////////////
        else if(strcmp(p,"#ifdef")==0)
        {
            layer++;
            p = strtok(NULL," \t\n");
            if(!p || *p==':')
            {
                printf("ERROR: incompleted command [#ifdef] in [%s:%d].\n", fname,line_id);
                exit(1);
            }

            int ikey;
            for(ikey=0; ikey<nkey; ikey++) if(strcmp(p,key[ikey])==0) break;
            if(ikey==nkey)
            {
                ignore = 1;
                fate = 0;
            }
        }

        //////////////////////////////////////////
        // #else
        //////////////////////////////////////////
        else if(strcmp(p,"#else")==0)
        {
            ignore = 1;
            fate = 0;
        }

        //////////////////////////////////////////
        // #endif
        //////////////////////////////////////////
        else if(strcmp(p,"#endif")==0)
        {
            if(layer<0)
            {
                printf("ERROR: unmatched [#if] and[#endif] at [%s:%d].\n",fname,line_id);
                exit(1);
            }
        }

        //////////////////////////////////////////
        // data
        //////////////////////////////////////////
        else
        {
            while(true)
            {
                char *src = p;
                for(int i=0; i<nkey; i++) if(strcmp(p,key[i])==0 && key_content[i])
                {
                    src = key_content[i];
                    break;
                }

                int len = strlen(src);
                while(nch+len+1>=buf_size)
                {
                    buf_size+=INC_BUF;
                    buf = (char*) realloc(buf, sizeof(char)*buf_size);
                }

                word[nword] = nch;
                strcpy(buf+nch,src);
                buf[nch+len]=0;
                nch += (len+1);
                nword ++;
                
                p = strtok(NULL," \t\n");
                if(!p || *p==':') break;
            }
        }

        // end of parse
    }

    if(level == 1) {
      fprintf(fp_cfg,"RAPTOR cfg file parse completed.\n");
      fprintf(fp_cfg,"NOTE: For boolean values, 0=False and 1=True\n");
      fprintf(fp_cfg, "\nNumber of unique keys: nkey= %i\n",nkey);
      for(int i=0; i<nkey; i++) fprintf(fp_cfg,"i= %3i  key= %30s  key_content= %s\n",i,key[i],key_content[i]);
    }

    level--;
}

/* ------------------------------------------------------------------ */
