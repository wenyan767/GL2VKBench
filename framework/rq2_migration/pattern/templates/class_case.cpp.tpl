#include "tcuDefs.hpp"
#include "tcuTestCase.hpp"
#include "tcuTestLog.hpp"
#include "tcuCommandLine.hpp"
#include "tcuTexture.hpp"
#include "tcuTextureUtil.hpp"
#include "tcuImageCompare.hpp"
#include "tcuVector.hpp"
#include "tcuVectorUtil.hpp"
#include "tcuVectorType.hpp"
#include "tcuStringTemplate.hpp"
#include "tcuResultCollector.hpp"
#include "tcuPlatform.hpp"
#include "tcuFormatUtil.hpp"
#include "tcuMaybe.hpp"
#include "tcuRGBA.hpp"
#include "tcuFloat.hpp"
#include "tcuResource.hpp"

#include "deDefs.h"
#include "deDefs.hpp"
#include "deMemory.h"
#include "deMath.h"
#include "deString.h"
#include "deStringUtil.hpp"
#include "deSTLUtil.hpp"
#include "deUniquePtr.hpp"
#include "deSharedPtr.hpp"
#include "deRandom.hpp"

#include "vkDefs.hpp"
#include "vkRef.hpp"
#include "vkRefUtil.hpp"
#include "vkPrograms.hpp"
#include "vkCmdUtil.hpp"
#include "vkTypeUtil.hpp"
#include "vkObjUtil.hpp"
#include "vkQueryUtil.hpp"
#include "vkBuilderUtil.hpp"
#include "vkImageUtil.hpp"
#include "vkBarrierUtil.hpp"
#include "vkMemUtil.hpp"
#include "vkBufferWithMemory.hpp"
#include "vkImageWithMemory.hpp"
#include "vkPlatform.hpp"
#include "vkDeviceUtil.hpp"
#include "vkStrUtil.hpp"

#include "vktTestCase.hpp"
#include "vktTestCaseUtil.hpp"
#include "vktTestGroupUtil.hpp"
#include "vktCustomInstancesDevices.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace vkt
{
namespace custom_migration
{

{{CLASS_CODE}}

void {{REGISTER_NAME}}(tcu::TestCaseGroup* group)
{
{{REGISTER_BODY}}
}

} // custom_migration
} // vkt
