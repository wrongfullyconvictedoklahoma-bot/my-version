import { Platform } from 'react-native';
import { OpenAILite } from '@/utils/AiProviders/openAILite';
import { applyTemplate } from 'chat-formatter';
import CactusLmWrapper, { NativeLlamaChatMessage } from '@/utils/AiProviders/onDevice/cactus';
import { ICompleteResponse } from '@/utils/AiProviders/baseOpenAILikeProvider';
import { LLMProvider } from '@/utils/AiProviders/baseLLMProvider';
import { Model } from '@/utils/types';
import { defaultModels } from '@/utils/models';

export interface OnDeviceProviderConstructorProps {
  config: {
    model?: string;
  };
  workspace?: any;
}

export default class OnDeviceProvider extends LLMProvider {
  private static instance: OnDeviceProvider;
  private config: { model?: string };
  private model: string | null = null;
  private submodule: CactusLmWrapper | null = null;
  private workspace: any;
  private provider = 'native';

  constructor({ config }: OnDeviceProviderConstructorProps) {
    super({ provider: 'native', config });

    // For compliance with the base class - we stub it here.
    this.client = new OpenAILite();
    this.isOTypeModel = false;
    this.temperature = 0.7;

    this.provider = 'native';
    this.config = config;
    this.model = this.config.model;

    if (this.model) {
      this.submodule = this.setSubmodule(this.model);
      this.log(`${this.name}::${this.submodule.name} initialized with model ${this.model}`);
    }
  }

  log = (text: string, ...args: any[]) => {
    console.log(`\x1b[36m[${this.constructor.name}:${this.submodule?.name || 'no-model'}]\x1b[0m ${text}`, ...args);
  }

  private setSubmodule(model: string) {
    if (!model) throw new Error('No model provided to setSubmodule');
    return new CactusLmWrapper({ model, parent: this });
  }

  static getInstance(props: OnDeviceProviderConstructorProps) {
    if (!OnDeviceProvider.instance) OnDeviceProvider.instance = new OnDeviceProvider(props);
    return OnDeviceProvider.instance;
  }

  /**
   * Delegates to the submodule to cleanup the model.
   */
  async unloadModel() {
    if (this.submodule) {
      await this.submodule.cleanup();
    }
  }

  get name() {
    return this.provider;
  }

  async loadNewModel(model: string | null) {
    if (!model) {
      this.log('No model provided to loadNewModel - cleaning up.');
      if (this.submodule) {
        await this.submodule.cleanup();
        this.submodule = null;
      }
      this.model = null;
      return;
    }

    if (this.model === model) return;
    this.model = model;
    if (this.submodule) {
      await this.submodule.cleanup();
    }
    this.submodule = this.setSubmodule(this.model);
    this.log(`${this.name}::${this.submodule.name} re-initialized with model ${this.model}`);
  }

  // @ts-ignore
  override async availableModels(): Promise<object[]> {
    return defaultModels.map(m => ({
      id: m.id,
      name: m.name,
      description: m.description,
      size: m.size,
      modelId: m.id,
      downloadUrl: m.downloadUrl || '',
      isPreset: false,
      imageUrl: m.imageUrl ?? null,
    }));
  }

  async runBasicChatCompletion(messages: any[]): Promise<ICompleteResponse> {
    return this.submodule!.getChatCompletion(messages);
  }

  override async chat({
    messages,
    streaming = false,
    onComplete = () => { },
    onStream = () => { },
  }: {
    messages: { role: string; content: string }[];
    streaming?: boolean;
    onComplete?: (response: ICompleteResponse) => void;
    onStream?: (token: string) => void;
  }): Promise<ICompleteResponse> {
    if (!this.submodule) throw new Error('No model loaded');

    const formattedMessages: NativeLlamaChatMessage[] = messages.map(m => ({
      role: m.role,
      content: m.content,
    }));

    if (streaming) {
      return this.submodule.streamGetChatCompletion(formattedMessages, onStream, []);
    } else {
      return this.submodule.getChatCompletion(formattedMessages);
    }
  }
}
