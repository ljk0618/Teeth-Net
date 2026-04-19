1. 牙根尖周炎（Periapical Periodontitis, PAP）是指发生于牙根尖周围组织的炎性病变，通常与牙髓感染向根尖区域扩散有关，在口腔全景片中多表现为牙根尖周围局部透射影增大或周围骨质稀疏。例如，下图中右下第一磨牙牙根尖周围存在明显透射影增大，则视为发生牙根尖周炎病变。标注时，采用 Labelme 软件创建多边形，沿病灶可见边界逐点勾画，对透射影异常区域进行完整标注；当病灶边界存在轻度模糊时，以异常透射影的主要外缘作为标注边界。
<img width="2276" height="1152" alt="api_83" src="https://github.com/user-attachments/assets/7d5f8af8-6992-4851-9d6c-0659d26a9c2d" />
<img width="2139" height="1137" alt="image" src="https://github.com/user-attachments/assets/ae3a4367-2289-4341-9c87-3be136b6a039" />
2. 龋齿（Dental Caries, DC）是牙体硬组织在细菌作用下发生的破坏性病变，通常由牙菌斑长期作用及酸性代谢产物侵蚀所致。在口腔全景片中，多表现为牙冠或邻面区域局部灰度减低，严重时可伴随牙体组织缺损。例如，下图中某磨牙邻面区域出现局部低密度透射影，则视为龋齿病变。标注时，采用 Labelme 软件创建多边形，沿龋坏区域的实际边界进行勾画，以完整覆盖病变范围；当病变边界不清晰时，以灰度异常区域的主要外缘作为标注边界。
<img width="1984" height="1152" alt="dec_10" src="https://github.com/user-attachments/assets/3d75c6d4-d185-48a2-9000-8a41dc22ecd4" />
<img width="1959" height="1131" alt="image" src="https://github.com/user-attachments/assets/279b4d3f-22cd-49b2-a96f-9404a9ab5a19" />
3. 根分叉病变（Furcation Involvement, FI）是指牙根分叉区域牙周支持组织发生吸收或破坏的病变，通常与牙周炎进展导致的牙槽骨丧失有关。在口腔全景片中，多表现为根分叉区骨质密度降低或局部透射影增大。例如，下图中某磨牙根分叉区可见明显骨质稀疏或透射影异常，则视为根分叉病变。标注时，采用 Labelme 软件创建多边形，沿根分叉区域异常骨质吸收范围进行勾画，以覆盖主要病变区域；对于边界模糊的情况，以分叉区灰度异常的主要外缘作为标注依据。
<img width="2280" height="1152" alt="fui_196" src="https://github.com/user-attachments/assets/245b40b6-b0bb-4bd0-86fd-3a1c530279e7" />
<img width="2112" height="1068" alt="image" src="https://github.com/user-attachments/assets/9e8fe6ce-51b2-4389-af3a-67292b05de8b" />
4. 阻生齿（Impacted Tooth, IT）是指由于萌出空间不足、萌出方向异常或邻近组织阻挡而未能正常萌出的牙齿。在口腔全景片中，通常表现为牙体埋伏于颌骨内、倾斜萌出、水平阻生或与邻牙位置关系异常。例如，下图中第三磨牙呈水平位或近中倾斜位阻生，且未能正常萌出，则视为阻生齿。标注时，采用 Labelme 软件创建多边形，沿阻生齿牙冠及牙根的可见轮廓进行整体勾画，以完整覆盖阻生牙区域；若部分边界与周围骨质重叠，则以牙体可辨识外缘为主要标注依据。

