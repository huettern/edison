/*
* @Author: Noah Huetter
* @Date:   2020-04-13 13:51:28
* @Last Modified by:   Noah Huetter
* @Last Modified time: 2020-05-01 10:33:48
*/
#include "version.h"

#include "main.h"
#include "printf.h"

/*------------------------------------------------------------------------------
 * Publics
 * ---------------------------------------------------------------------------*/
/**
 * @brief Prints versino string to stdout
 * @details 
 */
void verPrint(void)
{
  printf("%s / %s / %s / %s\n", verProgName, verVersion, verBuildDate, verGitSha);
}