import { registerComponent, PluginComponentType } from "@fiftyone/plugins";
import PerceptronChatPanel from "./PerceptronChatPanel";

/**
 * Register the PerceptronChatPanel React component.
 *
 * The ``name`` must match the ``component`` kwarg in
 * ``PerceptronChatPanel.render()`` in ``chat_panel.py``.
 */
registerComponent({
  name: "PerceptronChatPanel",
  component: PerceptronChatPanel,
  type: PluginComponentType.Component,
});
