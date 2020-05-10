
#include "ai_nnom.h"

#include "nnom.h"
#include "kws_nnom/weights.h"

void aiNnomTest(void)
{
#ifdef NNOM_VERIFICATION
  nnom_model_t *model;
  model = nnom_model_create();
  model_run(model);
#endif
}